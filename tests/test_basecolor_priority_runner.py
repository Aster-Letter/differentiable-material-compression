from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from scripts import train_scifihelmet_c4_basecolor_priority_10k as runner
from scripts import train_complex_c4_basecolor_priority_10k as generic_runner
from scripts import render_scifihelmet_c4_basecolor_priority_summary as visual_runner
from scripts import render_complex_c4_basecolor_priority_summary as complex_visual_runner


CONFIG = Path("configs/train/scifihelmet_c4_basecolor_priority_10k_v1.yaml")
ORACLE_CONFIG = Path("configs/train/scifihelmet_c4_basecolor_only_oracle_10k_v1.yaml")


def test_scifihelmet_runner_has_no_script_level_import_dependencies() -> None:
    runner_path = Path("scripts/train_scifihelmet_c4_basecolor_priority_10k.py")
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    script_modules = {path.stem for path in Path("scripts").glob("*.py")}
    dependencies = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in script_modules
    }

    assert dependencies == set()


def test_scifihelmet_visual_summary_only_uses_bundled_runner_dependency() -> None:
    summary_path = Path("scripts/render_scifihelmet_c4_basecolor_priority_summary.py")
    tree = ast.parse(summary_path.read_text(encoding="utf-8"))
    script_modules = {path.stem for path in Path("scripts").glob("*.py")}
    dependencies = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in script_modules
    }

    assert dependencies == {"train_scifihelmet_c4_basecolor_priority_10k"}


def test_checked_in_basecolor_candidate_matrix_and_budget_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert runner._checkpoint_steps(config["training"]) == frozenset({1000, 5000, 10000})
    assert not {
        "adaptive_profile",
        "adaptive_profile_sha256",
        "basecolor_profile_hash",
        "visibility_hash",
        "standard_p0_manifest",
        "standard_p0_manifest_sha256",
    }.intersection(config["source"])
    assert config["training"]["latent_learning_rate"] == pytest.approx(2.0e-4)
    assert config["training"]["affine_learning_rate"] == pytest.approx(2.0e-5)
    assert "support_penalty_weight" not in config["training"]
    assert "support" not in config["loss"]
    assert config["loss"] == {
        "render_linear": 1.0,
        "render_log": 0.25,
        "normal_cosine": 0.25,
        "roughness_l1": 0.5,
        "metallic_l1": 0.5,
        "charbonnier_epsilon": 0.001,
    }
    assert runner._candidate_spec(config, "N0-control").objective_id == "r0_control"
    assert runner._candidate_spec(config, "BC80").target_share == pytest.approx(0.8)
    assert runner._candidate_spec(config, "BC90").target_share == pytest.approx(0.9)
    assert runner._candidate_spec(config, "BC80-compander").compander is True
    assert runner._candidate_spec(config, "BC90-compander").compander is True


def test_basecolor_only_oracle_is_isolated_from_the_frozen_gate_matrix() -> None:
    gate = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    oracle = yaml.safe_load(ORACLE_CONFIG.read_text(encoding="utf-8"))

    assert "BC-only" not in gate["candidates"]
    candidate = runner._candidate_spec(oracle, "BC-only")
    assert candidate.objective_id == "basecolor_only_oracle"
    assert candidate.target_share == pytest.approx(1.0)
    assert candidate.compander is False
    assert runner._audit_target({}, candidate) == (
        1.0,
        {"latent": 1.0, "affine": 1.0},
    )


@pytest.mark.asset
def test_scifihelmet_runner_reloads_the_frozen_raw_parent_bytes() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    parent = runner._load_frozen_raw_parent(config["source"])

    assert parent.artifact_id == "p0-raw"
    assert parent.artifact_hash == config["source"]["raw_parent_artifact_hash"]
    assert parent.latent_unorm8.shape == (2048, 2048, 4)
    assert parent.weight.shape == (7, 4)
    assert parent.bias.shape == (7,)
    assert parent.certificate is None


@pytest.mark.parametrize(
    "schedule",
    [[], [1000, 1000, 10000], [5000, 1000, 10000], [1000, 5000]],
)
def test_checkpoint_schedule_fails_closed(schedule: list[int]) -> None:
    with pytest.raises(ValueError):
        runner._checkpoint_steps({"steps": 10000, "checkpoint_steps": schedule})


def test_generic_runner_only_accepts_assets_selected_by_phase_zero() -> None:
    config = yaml.safe_load(
        Path("configs/train/complex_c4_basecolor_priority_10k_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "status": "complete_two_validators_selected",
        "selection": {
            "basecolor_dominant": "BoomBox",
            "cross_channel_dominant": "Corset",
        },
    }

    reviews = {row["id"]: row.get("emissive_identity_preserved") for row in config["assets"]}
    assert reviews == {"BoomBox": True, "Corset": None, "WaterBottle": True}

    assert generic_runner._selected_spec(config, summary, "BoomBox")["id"] == "BoomBox"
    assert generic_runner._selected_spec(config, summary, "Corset")["id"] == "Corset"
    with pytest.raises(ValueError):
        generic_runner._selected_spec(config, summary, "WaterBottle")
    with pytest.raises(ValueError):
        generic_runner._selected_spec(config, {"status": "stopped_no_two_eligible_assets"}, "BoomBox")


def test_scow_scripts_lock_resources_serial_queue_and_job_scoped_outputs() -> None:
    formal = Path("scripts/scow_c4_basecolor_priority_job.slurm").read_text(encoding="utf-8")
    submit = Path("scripts/scow_submit_c4_basecolor_priority.sh").read_text(encoding="utf-8")
    remote = Path("scripts/remote_run_c4_basecolor_priority.sh").read_text(encoding="utf-8")

    assert "#SBATCH --partition=Students" in formal
    assert "#SBATCH --gres=gpu:RTX5090:1" in formal
    assert "#SBATCH --cpus-per-task=4" in formal
    assert "#SBATCH --time=04:00:00" in formal
    assert "logs/slurm/c4-basecolor-priority/%x.%j" in formal
    assert "--mem" not in formal and "--account" not in formal and "--qos" not in formal
    assert 'squeue -h -u "$USER" -t PENDING,RUNNING' in submit
    assert 'sbatch "$SCRIPT" "$@"' in submit
    assert 'outputs/remote/c4-basecolor-priority/${SLURM_JOB_ID}' in remote
    assert "Refusing to run on login node" in remote
    assert '"$@"' in remote
    assert "helmet-summary)" in remote
    assert "complex-summary)" in remote


def test_all_helmet_output_modes_remain_inside_repository() -> None:
    repository_output = runner.ROOT / "outputs" / "basecolor-path-contract"
    assert runner._repo_output_path(repository_output).is_relative_to(runner.ROOT)
    assert visual_runner._repo_output_path(repository_output).is_relative_to(
        visual_runner.ROOT
    )
    outside = runner.ROOT.parent / "c4-output-escape"
    with pytest.raises(ValueError):
        runner._repo_output_path(outside)
    with pytest.raises(ValueError):
        visual_runner._repo_output_path(outside)


def test_visual_summary_uses_frozen_configured_views() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    views = visual_runner._visual_views(config)

    assert [value[0] for value in views] == [
        "front",
        "upper_side",
        "rear_pipeline",
        "top",
    ]
    assert views[0][1:4] == (0, 0, True)
    assert views[1][1:4] == (9, 4, True)
    assert views[2][1:4] == (20, 2, False)
    assert views[3][1] is None and views[3][3] is False


def test_complex_visual_summary_requires_control_selected_and_compander() -> None:
    root = complex_visual_runner.ROOT
    candidates, selected = complex_visual_runner._candidate_matrix(
        [
            f"N0-control={root}",
            f"BC90={root}",
            f"BC90-compander={root}",
        ]
    )

    assert selected == "BC90"
    assert set(candidates) == {"N0-control", "BC90", "BC90-compander"}
    with pytest.raises(ValueError):
        complex_visual_runner._candidate_matrix(
            [f"N0-control={root}", f"BC90={root}"]
        )
    with pytest.raises(ValueError):
        complex_visual_runner._candidate_matrix(
            [
                f"N0-control={root}",
                f"BC80={root}",
                f"BC90-compander={root}",
            ]
        )
