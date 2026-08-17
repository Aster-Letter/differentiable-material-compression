from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from audit_scifihelmet_affine_render_sampling import (  # noqa: E402
    build_sample_bundle,
    camera_light_alignment,
    evaluate_camera_relative_audit,
    masked_reference_statistics,
    plan_sample_pairs,
)


def test_audit_replays_each_batch_with_independent_camera_and_light_draws() -> None:
    records = plan_sample_pairs(
        [[62, 17, 999], [30, 18, 998]],
        camera_names=[f"camera-{index}" for index in range(31)],
        light_names=[f"light-{index}" for index in range(6)],
        first_step=80_001,
    )

    assert records == [
        {
            "step": 80_001,
            "camera_draw": 62,
            "light_draw": 17,
            "camera_index": 0,
            "light_index": 5,
            "camera_name": "camera-0",
            "light_name": "light-5",
        },
        {
            "step": 80_002,
            "camera_draw": 30,
            "light_draw": 18,
            "camera_index": 30,
            "light_index": 0,
            "camera_name": "camera-30",
            "light_name": "light-0",
        },
    ]


def test_audit_reports_camera_relative_light_hemisphere() -> None:
    front = camera_light_alignment(
        camera_eye=(0.0, 0.0, 4.0),
        light_position=(0.0, 0.0, 3.0),
        target=(0.0, 0.0, 0.0),
    )
    rear = camera_light_alignment(
        camera_eye=(0.0, 0.0, 4.0),
        light_position=(0.0, 0.0, -3.0),
        target=(0.0, 0.0, 0.0),
    )

    assert math.isclose(front["cosine"], 1.0)
    assert math.isclose(front["angle_degrees"], 0.0)
    assert front["hemisphere"] == "camera_side"
    assert math.isclose(rear["cosine"], -1.0)
    assert math.isclose(rear["angle_degrees"], 180.0)
    assert rear["hemisphere"] == "opposite_side"


def test_audit_exports_individual_renders_and_a_labeled_contact_sheet() -> None:
    samples = [
        {
            "step": 80_001,
            "camera_name": "camera-a",
            "light_name": "light-a",
            "display_rgb": np.full((4, 5, 3), 32, dtype=np.uint8),
        },
        {
            "step": 80_002,
            "camera_name": "camera-b",
            "light_name": "light-b",
            "display_rgb": np.full((4, 5, 3), 224, dtype=np.uint8),
        },
    ]

    manifest, files = build_sample_bundle(samples, columns=2)

    assert manifest["sample_count"] == 2
    assert manifest["contact_sheet"]["size"] == [10, 28]
    assert set(files) == {
        "contact_sheet.png",
        "sample-080001-camera-a__light-a.png",
        "sample-080002-camera-b__light-b.png",
    }
    assert all(payload.startswith(b"\x89PNG\r\n\x1a\n") for payload in files.values())


def test_audit_reports_masked_reference_luminance_without_background_bias() -> None:
    hdr = np.array(
        [
            [[0.1, 0.1, 0.1], [99.0, 99.0, 99.0]],
            [[1.0, 1.0, 1.0], [99.0, 99.0, 99.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[True, False], [True, False]])

    statistics = masked_reference_statistics(hdr, mask, dark_threshold=0.2)

    assert statistics["valid_pixels"] == 2
    assert np.allclose(statistics["mean_rgb"], [0.55, 0.55, 0.55])
    assert math.isclose(statistics["luminance_mean"], 0.55, rel_tol=1e-6)
    assert math.isclose(statistics["luminance_p50"], 0.55, rel_tol=1e-6)
    assert statistics["dark_fraction"] == 0.5


def test_camera_relative_audit_enforces_frozen_hemisphere_and_visibility_gates() -> None:
    records = []
    for camera_index in range(31):
        for light_index in range(6):
            is_rim = light_index == 5
            records.append(
                {
                    "camera_index": camera_index,
                    "light_index": light_index,
                    "role": "rim" if is_rim else "camera_side",
                    "hemisphere": "opposite_side" if is_rim else "camera_side",
                    "reference": {"dark_fraction": 0.65 if not is_rim else 0.95},
                    "direct_light": {"mean_positive": 0.40 if not is_rim else 0.02},
                }
            )

    audit = evaluate_camera_relative_audit(records, camera_count=31, light_count=6)

    assert audit["weighted_opposite_probability"] == pytest.approx(1.0 / 6.0)
    assert audit["non_rim_opposite_count"] == 0
    assert audit["non_rim_mean_dark_fraction"] == pytest.approx(0.65)
    assert audit["non_rim_mean_positive_n_dot_l"] == pytest.approx(0.40)
    assert audit["gates_passed"] is True
