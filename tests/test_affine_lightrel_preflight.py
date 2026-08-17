from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cg_frontier.render.pbr import PointLight
from run_scifihelmet_c4_affine_preflight import PreflightObjective


def _light(x: float) -> PointLight:
    return PointLight(
        position=(x, 0.0, 1.0),
        color=(1.0, 1.0, 1.0),
        radiant_intensity=1.0,
        ambient_intensity=0.0,
    )


def test_preflight_resolves_camera_specific_light_without_changing_family_draw() -> None:
    objective = object.__new__(PreflightObjective)
    objective.helmet_lights = [_light(10.0), _light(20.0)]
    objective.helmet_light_grid = [
        [_light(100.0), _light(101.0)],
        [_light(200.0), _light(201.0)],
    ]

    assert objective.resolve_helmet_light(camera_index=1, light_index=0).position == (
        101.0,
        0.0,
        1.0,
    )
    assert objective.resolve_helmet_light(camera_index=0, light_index=1).position == (
        200.0,
        0.0,
        1.0,
    )


def test_preflight_reports_the_selected_render_pair_without_changing_it() -> None:
    observed: list[tuple[int, int]] = []
    objective = object.__new__(PreflightObjective)
    objective.on_render_pair = lambda camera_index, light_index: observed.append(
        (camera_index, light_index)
    )

    assert objective.observe_render_pair(camera_index=7, light_index=4) == (7, 4)
    assert observed == [(7, 4)]

    objective.on_render_pair = None
    assert objective.observe_render_pair(camera_index=2, light_index=5) == (2, 5)
    assert observed == [(7, 4)]
