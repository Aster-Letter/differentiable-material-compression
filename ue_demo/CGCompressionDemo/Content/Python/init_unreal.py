"""Project Python startup hooks, all disabled unless explicitly requested."""

from __future__ import annotations

import os
from pathlib import Path
import runpy

import unreal


if os.environ.get("UE_RUNTIME_CAPTURE_AUTOSTART") == "1":
    capture_script = Path(__file__).with_name("capture_ue_runtime_residency_v1.py")
    unreal.log(f"Starting guarded UE runtime residency capture: {capture_script}")
    runpy.run_path(str(capture_script), run_name="__ue_runtime_residency_v1__")

if os.environ.get("UE_BC_LATENT_CAPTURE_AUTOSTART") == "1":
    capture_script = Path(__file__).with_name("capture_ue_bc_latent_residency_v1.py")
    unreal.log(f"Starting guarded BC latent residency capture: {capture_script}")
    runpy.run_path(str(capture_script), run_name="__ue_bc_latent_residency_v1__")

if os.environ.get("UE_BC_LATENT_VISUAL_EDITOR_AUTOSTART") == "1":
    capture_script = Path(__file__).with_name("capture_ue_bc_latent_visual_editor_v1.py")
    unreal.log(f"Starting guarded BC latent editor visual capture: {capture_script}")
    runpy.run_path(str(capture_script), run_name="__ue_bc_latent_visual_editor_v1__")
