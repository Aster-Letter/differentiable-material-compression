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

from cg_frontier.compression.ue_hybrid_direct_scalars_export import (  # noqa: E402
    ARRAYS,
    HLSL_NAME,
    MANIFEST_NAME,
    TEXTURE_A_NAME,
    TEXTURE_B_NAME,
    export,
    generate_hlsl,
    parse_hlsl,
    validate,
)


def test_frozen_r0b_contract_and_cost() -> None:
    a, b, arrays, metadata = validate(ROOT)
    assert a.shape == (2048, 2048, 4)
    assert b.shape == (2048, 2048, 3)
    assert set(arrays) == set(ARRAYS)
    assert (metadata["parameter_count"], metadata["weight_bytes_float32"], metadata["macs_per_pixel"]) == (32, 128, 24)


def test_generated_hlsl_round_trip_and_direct_paths() -> None:
    _, _, arrays, metadata = validate(ROOT)
    source = generate_hlsl(arrays, metadata)
    parsed = parse_hlsl(source)
    for name in parsed:
        np.testing.assert_array_equal(parsed[name], arrays[name])
    assert "return TextureA.rgb;" in source
    assert "Roughness = TextureB.g;" in source
    assert "Metallic = TextureB.b;" in source
    assert "sigmoid" not in source.lower()


def test_export_package_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export(ROOT, first)
    export(ROOT, second)
    for name in (HLSL_NAME, TEXTURE_A_NAME, TEXTURE_B_NAME, "probe_vectors.json", "UE_INTEGRATION_CHECKLIST.md", MANIFEST_NAME):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["formal_holdout_accessed"] is False
    assert manifest["budget"]["ue_physical_ceiling_bytes"] == 33554432
