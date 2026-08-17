from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.deployment_parity import DeploymentParityDecoder  # noqa: E402
from cg_frontier.render.canonical_v2 import (  # noqa: E402
    CanonicalRendererV2,
    LatentMaterialSource,
    ReferenceMaterialSource,
    compare_render_bundles,
    compare_gradient_probes,
    parity_report_json,
)
from cg_frontier.render.gbuffer import Camera, GBufferResult, MaterialBuffers  # noqa: E402
from cg_frontier.render.pbr import PointLight  # noqa: E402


def _geometry() -> GBufferResult:
    mask = torch.tensor([[True]])
    return GBufferResult(
        buffers={},
        torch_buffers={
            "mask": mask,
            "uv": torch.tensor([[[0.5, 0.5]]], dtype=torch.float32),
            "position_world": torch.zeros((1, 1, 3), dtype=torch.float32),
            "vertex_normal_world": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "tangent_world": torch.tensor([[[1.0, 0.0, 0.0]]]),
            "bitangent_world": torch.tensor([[[0.0, 1.0, 0.0]]]),
        },
        metadata={"fixture": "one_pixel"},
    )


def _camera() -> Camera:
    return Camera(
        eye=(0.0, 0.0, 4.5),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        vertical_fov_degrees=45.0,
        near=0.1,
        far=10.0,
    )


def _light() -> PointLight:
    return PointLight(
        position=(2.5, 3.0, 4.0),
        color=(1.0, 0.98, 0.95),
        radiant_intensity=90.0,
        ambient_intensity=0.04,
    )


def _black_hole_source() -> LatentMaterialSource:
    latent = torch.zeros((2, 2, 4), dtype=torch.float32)
    latent[..., 0] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    decoder = DeploymentParityDecoder(width=8)
    with torch.no_grad():
        decoder.hidden.weight.zero_()
        decoder.hidden.bias.fill_(-1.0)
        decoder.output.weight.zero_()
        decoder.output.bias.zero_()
        decoder.hidden.weight[0, 0] = 1.0
        decoder.hidden.bias[0] = -0.75
        decoder.hidden.weight[1, 0] = -1.0
        decoder.hidden.bias[1] = 0.25
        decoder.output.bias[0:3] = -6.0
        decoder.output.weight[0:3, 0:2] = 48.0
    return LatentMaterialSource(latent, decoder, quantization="hard")


def test_canonical_renderer_v2_returns_the_complete_deployment_bundle() -> None:
    renderer = CanonicalRendererV2(display_exposure=1.0, minimum_roughness=0.045)
    bundle = renderer.render(
        _geometry(),
        _camera(),
        _light(),
        _black_hole_source(),
        input_hashes={"fixture": "sha256:test"},
    )

    assert bundle.renderer_identifier == "canonical_renderer_v2"
    assert torch.equal(bundle.coverage, torch.tensor([[True]]))
    assert torch.equal(bundle.uv, torch.tensor([[[0.5, 0.5]]]))
    assert torch.max(bundle.material.base_color_linear) < 0.01
    assert bundle.linear_hdr.shape == (1, 1, 3)
    assert torch.isfinite(bundle.linear_hdr).all()
    assert torch.isfinite(bundle.display_rgb).all()
    assert bundle.metadata["deployment_order"] == [
        "rgba8_quantize_texels",
        "single_bilinear_latent_sample",
        "relu_decoder_4_to_8_to_7",
        "core4_postprocess_once",
        "shared_ggx_pbr",
        "display_transform",
    ]
    assert bundle.metadata["input_hashes"] == {"fixture": "sha256:test"}


def test_reference_branch_and_parity_report_are_deterministic() -> None:
    geometry = _geometry()
    material = MaterialBuffers(
        base_color_linear=torch.tensor([[[0.2, 0.4, 0.6]]]),
        normal_world=torch.tensor([[[0.0, 0.0, 1.0]]]),
        roughness=torch.tensor([[0.4]]),
        metallic=torch.tensor([[0.7]]),
        normal_ts_raw=torch.tensor([[[0.0, 0.0, 1.0]]]),
        normal_ts_unit=torch.tensor([[[0.0, 0.0, 1.0]]]),
    )
    renderer = CanonicalRendererV2(display_exposure=1.0, minimum_roughness=0.045)
    first = renderer.render(
        geometry,
        _camera(),
        _light(),
        ReferenceMaterialSource(material),
        input_hashes={"fixture": "sha256:reference"},
    )
    second = renderer.render(
        geometry,
        _camera(),
        _light(),
        ReferenceMaterialSource(material),
        input_hashes={"fixture": "sha256:reference"},
    )
    report = compare_render_bundles(first, second)

    assert report.passed
    assert report.metrics["coverage_exact"] is True
    assert report.metrics["uv_max_abs"] == 0.0
    assert report.metrics["display_ssim"] == 1.0
    assert parity_report_json(report) == parity_report_json(report)
    assert parity_report_json(report).endswith(b"\n")


def test_fixed_gradient_probe_gate_reports_cosine_and_relative_l2() -> None:
    generator = torch.Generator().manual_seed(20260804)
    probes = torch.rand((32, 4), generator=generator)
    weights = torch.tensor([[1.0, 2.0, -0.5, 0.25], [-0.2, 0.4, 0.8, 1.1]])

    report = compare_gradient_probes(
        lambda value: value @ weights.T,
        lambda value: value @ weights.T,
        probes,
    )

    assert report["probe_count"] == 32
    assert report["all_finite"] is True
    assert report["minimum_cosine"] >= 0.999
    assert report["maximum_relative_l2"] <= 0.01
    assert report["passed"] is True
