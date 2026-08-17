from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_scifihelmet_affine_lightrel import freeze_lightrel_audit_config


def test_lightrel_audit_freezes_full_31_by_6_reference_matrix() -> None:
    pool = {
        "render": {"resolution": [256, 256]},
        "train_cameras": [{"name": f"camera-{index}"} for index in range(31)],
    }
    config = {
        "experiment": "scifihelmet_c4_affine_chroma8_l0_lightrel_5k",
        "audit_gates": {"require_all_31x6_pairs": True},
        "camera_relative_lights": [
            {
                "name": f"light-{index}",
                "local_position": [0.0, 2.0, -3.0 if index == 5 else 3.0],
                "color": [1.0, 1.0, 1.0],
                "radiant_intensity": 80.0,
                "ambient_intensity": 0.04,
                "role": "rim" if index == 5 else "camera_side",
            }
            for index in range(6)
        ],
    }

    frozen = freeze_lightrel_audit_config(config, pool)

    assert frozen.resolution == (256, 256)
    assert frozen.camera_count == 31
    assert frozen.light_count == 6
    assert frozen.pair_count == 186
    assert frozen.camera_side_count == 5
    assert frozen.rim_count == 1
