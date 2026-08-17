from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_scifihelmet_c4_affine_color_guard import (
    certified_parent_hash,
    freeze_balanced_gate_policy,
    freeze_color_audit_spec,
)


def _config() -> dict[str, object]:
    return {
        "audit": {
            "batch_count": 8,
            "material_batch_size": 4096,
            "color_batch_size": 4096,
            "color_seed_offset": 23,
            "metric_seed_offset": 29,
            "pairs_per_bin_pair": 32,
            "quantiles": [0.25, 0.50, 0.75],
            "charbonnier_epsilon": 0.001,
            "gradient_epsilon": 1.0e-12,
            "ratios": [0.10, 0.25, 0.50],
            "selected_ratio": None,
        }
    }


def test_color_audit_spec_freezes_the_no_update_confirmation_point() -> None:
    spec = freeze_color_audit_spec(_config())

    assert spec.batch_count == 8
    assert spec.material_batch_size == 4096
    assert spec.color_batch_size == 4096
    assert spec.ratios == (0.10, 0.25, 0.50)
    assert spec.selected_ratio is None

    invalid = _config()
    invalid["audit"]["selected_ratio"] = 0.25
    with pytest.raises(ValueError, match="must not select"):
        freeze_color_audit_spec(invalid)


def test_balanced_gate_policy_never_uses_yellow_for_selection() -> None:
    policy = freeze_balanced_gate_policy()

    assert policy["parent_chroma_retention_floor"] == 0.90
    assert policy["global_error_multiplier"] == 1.05
    assert policy["color_improvement_vs_c0"] == 0.05
    assert policy["color_regression_vs_c0"] == 0.02
    assert policy["display_ssim_absolute_drop"] == 0.005
    assert policy["yellow_diagnostic"]["selection_metric"] is False


def test_certified_parent_hash_comes_from_safe_artifact_manifest() -> None:
    manifest = {
        "safe_artifact": {"artifact_id": "p0-safe", "artifact_hash": "safe-hash"},
        "raw_artifact": {"artifact_id": "p0-raw", "artifact_hash": "raw-hash"},
    }

    assert certified_parent_hash(manifest) == "safe-hash"

    with pytest.raises(ValueError, match="safe artifact"):
        certified_parent_hash({"artifact_hash": "wrong-level"})
