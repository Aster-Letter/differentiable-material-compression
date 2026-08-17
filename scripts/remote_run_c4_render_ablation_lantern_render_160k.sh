#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PROJECT_ROOT
ENV_ROOT="${ENV_ROOT:-$HOME/projects/cg_frontier/.envs/cg-frontier-system-py312}"
MODE="${1:-}"
shift || true
HOST_NAME="$(hostname)"

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Refusing to run outside Slurm." >&2; exit 2; }
case "$HOST_NAME" in *tradmin*|*login*) echo "Refusing to run on login node: $HOST_NAME" >&2; exit 2 ;; esac
[[ -x "$ENV_ROOT/bin/python" ]] || { echo "Missing environment Python: $ENV_ROOT/bin/python" >&2; exit 3; }
case "$MODE" in preflight|formal|resume) ;; *) echo "Usage: $0 <preflight|formal|resume> ..." >&2; exit 4 ;; esac

module load cuda/13.0
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$PROJECT_ROOT/.scratch/c4-render-ablation-lantern-render-160k-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
OUTPUT_BASE="outputs/remote/c4-render-ablation-lantern-render-160k-v1/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"

echo "job_id=$SLURM_JOB_ID"
echo "host=$HOST_NAME"
echo "project_root=$PROJECT_ROOT"
echo "mode=$MODE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV_ROOT/bin/python" -c 'from pathlib import Path; import os, cg_frontier; from cg_frontier.compression import render_ablation_long_continuation as m; expected=(Path(os.environ["PROJECT_ROOT"])/"src"/"cg_frontier").resolve(); assert Path(cg_frontier.__file__).resolve().parent == expected; assert Path(m.__file__).resolve().parent == expected/"compression"; print("campaign_package", cg_frontier.__file__, "long_continuation", m.__file__)'
"$ENV_ROOT/bin/python" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"

CONFIG="configs/train/c4_render_ablation_lantern_render_160k_v1.yaml"
SOURCE_MARKER="outputs/remote/c4-render-ablation-lantern-40k-v1/37581/formal_verified.json"
[[ -f "$SOURCE_MARKER" ]] || { echo "Missing verified 40k source marker: $SOURCE_MARKER" >&2; exit 5; }

if [[ "$MODE" == "preflight" ]]; then
  for script in \
    scripts/remote_run_c4_render_ablation_lantern_render_160k.sh \
    scripts/scow_c4_render_ablation_lantern_render_160k_preflight.slurm \
    scripts/scow_c4_render_ablation_lantern_render_160k_job.slurm \
    scripts/scow_submit_c4_render_ablation_lantern_render_160k.sh; do
    bash -n "$script"
  done
  "$ENV_ROOT/bin/python" scripts/continue_c4_render_ablation_lantern_render_160k.py \
    --config "$CONFIG" \
    --output-root "$OUTPUT_BASE/Lantern/material_render" \
    --max-continuation-steps 10
  "$ENV_ROOT/bin/python" scripts/verify_c4_render_ablation_lantern_render_160k_run.py \
    --mode preflight \
    --run-root "$OUTPUT_BASE" \
    --job-id "$SLURM_JOB_ID"
elif [[ "$MODE" == "formal" ]]; then
  PREFLIGHT_JOB_ID="${1:?formal mode requires the successful preflight job id}"
  MARKER="outputs/remote/c4-render-ablation-lantern-render-160k-v1/${PREFLIGHT_JOB_ID}/preflight_verified.json"
  [[ -f "$MARKER" ]] || { echo "Missing verified preflight marker: $MARKER" >&2; exit 5; }
  "$ENV_ROOT/bin/python" scripts/continue_c4_render_ablation_lantern_render_160k.py \
    --config "$CONFIG" \
    --output-root "$OUTPUT_BASE/Lantern/material_render"
  "$ENV_ROOT/bin/python" scripts/verify_c4_render_ablation_lantern_render_160k_run.py \
    --mode formal \
    --run-root "$OUTPUT_BASE" \
    --job-id "$SLURM_JOB_ID" \
    --preflight-job-id "$PREFLIGHT_JOB_ID"
else
  SOURCE_JOB_ID="${1:?resume mode requires the failed 160k job id}"
  CHECKPOINT_STEP="${2:?resume mode requires checkpoint step 80000 or 120000}"
  PREFLIGHT_JOB_ID="${3:?resume mode requires the successful preflight job id}"
  case "$CHECKPOINT_STEP" in 80000|120000) ;; *) echo "Resume step must be 80000 or 120000" >&2; exit 4 ;; esac
  RESULT_ROOT="outputs/remote/c4-render-ablation-lantern-render-160k-v1/${SOURCE_JOB_ID}/Lantern"
  ARM_ROOT="$RESULT_ROOT/material_render"
  CHECKPOINT="$ARM_ROOT/checkpoints/step_$(printf '%06d' "$CHECKPOINT_STEP")/checkpoint.pt"
  [[ -f "$CHECKPOINT" ]] || { echo "Missing resume checkpoint: $CHECKPOINT" >&2; exit 6; }
  [[ -f "outputs/remote/c4-render-ablation-lantern-render-160k-v1/${PREFLIGHT_JOB_ID}/preflight_verified.json" ]] || { echo "Missing verified preflight marker." >&2; exit 5; }
  "$ENV_ROOT/bin/python" scripts/continue_c4_render_ablation_lantern_render_160k.py \
    --config "$CONFIG" \
    --output-root "$ARM_ROOT" \
    --resume-checkpoint "$CHECKPOINT"
  "$ENV_ROOT/bin/python" scripts/verify_c4_render_ablation_lantern_render_160k_run.py \
    --mode formal \
    --run-root "$OUTPUT_BASE" \
    --result-root "$RESULT_ROOT" \
    --job-id "$SLURM_JOB_ID" \
    --preflight-job-id "$PREFLIGHT_JOB_ID"
fi
