from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.asset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.ue_export import (
    CANONICAL_PROBES,
    CHECKLIST_FILENAME,
    DEFAULT_OUTPUT_RELATIVE_PATH,
    DECODER_RELATIVE_PATH,
    EXPECTED_ARRAYS,
    EXPECTED_DECODER_SHA256,
    EXPECTED_LATENT_SHA256,
    HLSL_FILENAME,
    LATENT_EXPORT_FILENAME,
    LATENT_RELATIVE_PATH,
    MACS_PER_PIXEL,
    MANIFEST_FILENAME,
    PARAMETER_COUNT,
    PROBES_FILENAME,
    UV_PROBES,
    WEIGHT_BYTES_FLOAT32,
    bilinear_sample_top_down_wrap,
    decoder_postprocess,
    decoder_raw_forward,
    export_frozen_ue_package,
    generate_custom_hlsl,
    parse_generated_hlsl_constants,
    sha256_file,
    validate_frozen_inputs,
)


def _validated_inputs():
    return validate_frozen_inputs(
        ROOT / LATENT_RELATIVE_PATH, ROOT / DECODER_RELATIVE_PATH
    )


def test_frozen_inputs_have_exact_hash_shape_and_cost_metadata() -> None:
    latent_u8, arrays, metadata = _validated_inputs()
    assert metadata["latent_sha256"] == EXPECTED_LATENT_SHA256
    assert metadata["decoder_sha256"] == EXPECTED_DECODER_SHA256
    assert latent_u8.shape == (2048, 2048, 4)
    assert latent_u8.dtype == np.uint8
    assert metadata["latent_raw_gpu_bytes_no_mips"] == 16 * 1024 * 1024
    assert metadata["parameter_count"] == PARAMETER_COUNT == 103
    assert metadata["weight_bytes_float32"] == WEIGHT_BYTES_FLOAT32 == 412
    assert metadata["macs_per_pixel"] == MACS_PER_PIXEL == 88
    assert {name: array.shape for name, array in arrays.items()} == EXPECTED_ARRAYS


def test_generated_hlsl_constants_round_trip_and_fixed_probes_are_exact() -> None:
    latent_u8, arrays, metadata = _validated_inputs()
    hlsl = generate_custom_hlsl(arrays, metadata)
    reloaded = parse_generated_hlsl_constants(hlsl)
    for name in EXPECTED_ARRAYS:
        np.testing.assert_array_equal(reloaded[name], arrays[name])

    fixed_latent = np.concatenate(
        (CANONICAL_PROBES, bilinear_sample_top_down_wrap(latent_u8, UV_PROBES)), axis=0
    )
    expected_raw = decoder_raw_forward(fixed_latent, arrays)
    actual_raw = decoder_raw_forward(fixed_latent, reloaded)
    np.testing.assert_array_equal(actual_raw, expected_raw)
    expected_post = decoder_postprocess(expected_raw)
    actual_post = decoder_postprocess(actual_raw)
    for name in expected_post:
        np.testing.assert_array_equal(actual_post[name], expected_post[name])

    assert "dot(LatentRGBA" in hlsl
    assert "CGC_NormalGLTF.y * NormalYSign" in hlsl
    assert hlsl.count("tanh(float2(CGC_R3, CGC_R4))") == 1
    assert "return CGC_BaseColorLinear;" in hlsl


def test_export_package_manifest_and_probe_payload() -> None:
    output_dir = ROOT / DEFAULT_OUTPUT_RELATIVE_PATH
    manifest = export_frozen_ue_package(ROOT, output_dir)
    assert manifest["selection"] == "pre_qat_hard_tiny_mlp"
    assert manifest["decoder"] == {
        "architecture": "4->8->7",
        "activation": "ReLU",
        "parameter_count": 103,
        "weight_bytes_float32": 412,
        "macs_per_pixel": 88,
        "raw_equation": "raw=W2*ReLU(W1*z+b1)+b2",
    }
    assert manifest["runtime"]["base_color_additional_srgb_decode"] is False
    assert manifest["runtime"]["ambient_occlusion"] == "excluded"
    assert manifest["ue_faithful_baseline"]["srgb"] is False
    assert manifest["ue_faithful_baseline"]["mips"] == "disabled"
    assert manifest["ue_faithful_baseline"]["normal_y_bridge"]["application_count"] == 1
    assert manifest["verification"]["generated_constant_reload_max_abs"] == 0.0

    for filename in (
        HLSL_FILENAME,
        LATENT_EXPORT_FILENAME,
        PROBES_FILENAME,
        CHECKLIST_FILENAME,
        MANIFEST_FILENAME,
    ):
        assert (output_dir / filename).is_file()
    assert sha256_file(output_dir / LATENT_EXPORT_FILENAME) == EXPECTED_LATENT_SHA256

    probes = json.loads((output_dir / PROBES_FILENAME).read_text(encoding="utf-8"))
    assert len(probes["canonical"]["latent_rgba"]) == len(CANONICAL_PROBES)
    assert len(probes["real_uv"]["uv_top_down_wrap"]) == len(UV_PROBES)
