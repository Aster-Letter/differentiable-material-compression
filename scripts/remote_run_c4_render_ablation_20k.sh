#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PROJECT_ROOT
ENV_ROOT="${ENV_ROOT:-$HOME/projects/cg_frontier/.envs/cg-frontier-system-py312}"
MODE="${1:-}"
shift || true
HOST_NAME="$(hostname)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run C4 render ablation outside a Slurm allocation." >&2
  exit 2
fi
case "$HOST_NAME" in
  *tradmin*|*login*) echo "Refusing to run on login node: $HOST_NAME" >&2; exit 2 ;;
esac
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  echo "Missing environment Python: $ENV_ROOT/bin/python" >&2
  exit 3
fi
if [[ "$MODE" != "preflight" && "$MODE" != "formal" && "$MODE" != "resume" ]]; then
  echo "Usage: remote_run_c4_render_ablation_20k.sh <preflight|formal|resume> ..." >&2
  exit 4
fi

module load cuda/13.0
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$PROJECT_ROOT/.scratch/c4-render-ablation-20k-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
OUTPUT_BASE="outputs/remote/c4-render-ablation-20k-v1/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"

echo "job_id=$SLURM_JOB_ID"
echo "host=$HOST_NAME"
echo "project_root=$PROJECT_ROOT"
echo "mode=$MODE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV_ROOT/bin/python" -c 'from pathlib import Path; import os; import cg_frontier; from cg_frontier.compression import render_ablation; expected=(Path(os.environ["PROJECT_ROOT"])/"src"/"cg_frontier").resolve(); package=Path(cg_frontier.__file__).resolve(); module=Path(render_ablation.__file__).resolve(); assert package.parent == expected, (package, expected); assert module.parent == expected/"compression", (module, expected); print("campaign_package", package, "render_ablation", module)'
"$ENV_ROOT/bin/python" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"

CONFIG="configs/train/c4_render_ablation_20k_v1.yaml"
if [[ "$MODE" == "preflight" ]]; then
  bash -n scripts/remote_run_c4_render_ablation_20k.sh
  bash -n scripts/scow_c4_render_ablation_20k_preflight.slurm
  bash -n scripts/scow_c4_render_ablation_20k_job.slurm
  bash -n scripts/scow_submit_c4_render_ablation_20k.sh
  for ASSET in Corset Lantern BoomBox; do
    "$ENV_ROOT/bin/python" scripts/train_c4_render_ablation_20k.py \
      --config "$CONFIG" \
      --asset "$ASSET" \
      --output-root "$OUTPUT_BASE/$ASSET" \
      --max-steps 10
  done
  "$ENV_ROOT/bin/python" scripts/verify_c4_render_ablation_run.py \
    --mode preflight \
    --run-root "$OUTPUT_BASE" \
    --job-id "$SLURM_JOB_ID"
elif [[ "$MODE" == "formal" ]]; then
  ASSET="${1:?formal mode requires one asset}"
  PREFLIGHT_JOB_ID="${2:?formal mode requires the successful preflight job id}"
  case "$ASSET" in Corset|Lantern|BoomBox) ;; *) echo "Unknown asset: $ASSET" >&2; exit 4 ;; esac
  if [[ ! -f "outputs/remote/c4-render-ablation-20k-v1/${PREFLIGHT_JOB_ID}/preflight_verified.json" ]]; then
    echo "Missing verified preflight marker for job $PREFLIGHT_JOB_ID" >&2
    exit 5
  fi
  "$ENV_ROOT/bin/python" scripts/train_c4_render_ablation_20k.py \
    --config "$CONFIG" \
    --asset "$ASSET" \
    --output-root "$OUTPUT_BASE/$ASSET"
  "$ENV_ROOT/bin/python" scripts/render_c4_render_ablation_summary.py \
    --config "$CONFIG" \
    --asset "$ASSET" \
    --pair-root "$OUTPUT_BASE/$ASSET" \
    --output-root "$OUTPUT_BASE/$ASSET-summary"
  "$ENV_ROOT/bin/python" scripts/verify_c4_render_ablation_run.py \
    --mode formal \
    --asset "$ASSET" \
    --run-root "$OUTPUT_BASE" \
    --job-id "$SLURM_JOB_ID" \
    --preflight-job-id "$PREFLIGHT_JOB_ID"
else
  ASSET="${1:?resume mode requires one asset}"
  ARM="${2:?resume mode requires material_only or material_render}"
  SOURCE_JOB_ID="${3:?resume mode requires the failed source job id}"
  PREFLIGHT_JOB_ID="${4:?resume mode requires the successful preflight job id}"
  case "$ASSET" in Corset|Lantern|BoomBox) ;; *) echo "Unknown asset: $ASSET" >&2; exit 4 ;; esac
  case "$ARM" in material_only|material_render) ;; *) echo "Unknown arm: $ARM" >&2; exit 4 ;; esac
  SOURCE_ROOT="outputs/remote/c4-render-ablation-20k-v1/${SOURCE_JOB_ID}/${ASSET}"
  RESUME_CHECKPOINT="$SOURCE_ROOT/$ARM/checkpoints/step_10000/checkpoint.pt"
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Resume requires the verified 10k checkpoint: $RESUME_CHECKPOINT" >&2
    exit 6
  fi
  "$ENV_ROOT/bin/python" scripts/train_c4_render_ablation_20k.py \
    --config "$CONFIG" \
    --asset "$ASSET" \
    --arm "$ARM" \
    --output-root "$SOURCE_ROOT/$ARM" \
    --resume "$RESUME_CHECKPOINT"
  if [[ "$ARM" == "material_only" ]]; then OTHER_ARM="material_render"; else OTHER_ARM="material_only"; fi
  if [[ ! -f "$SOURCE_ROOT/$OTHER_ARM/training_report.json" ]]; then
    if [[ -e "$SOURCE_ROOT/$OTHER_ARM" ]]; then
      echo "Other arm has an incomplete output and cannot be silently restarted: $SOURCE_ROOT/$OTHER_ARM" >&2
      exit 7
    fi
    "$ENV_ROOT/bin/python" scripts/train_c4_render_ablation_20k.py \
      --config "$CONFIG" \
      --asset "$ASSET" \
      --arm "$OTHER_ARM" \
      --output-root "$SOURCE_ROOT/$OTHER_ARM"
  fi
  "$ENV_ROOT/bin/python" scripts/train_c4_render_ablation_20k.py \
    --config "$CONFIG" \
    --asset "$ASSET" \
    --output-root "$SOURCE_ROOT" \
    --finalize-pair
  "$ENV_ROOT/bin/python" scripts/render_c4_render_ablation_summary.py \
    --config "$CONFIG" \
    --asset "$ASSET" \
    --pair-root "$SOURCE_ROOT" \
    --output-root "$OUTPUT_BASE/$ASSET-summary"
  "$ENV_ROOT/bin/python" scripts/verify_c4_render_ablation_run.py \
    --mode formal \
    --asset "$ASSET" \
    --pair-root "$SOURCE_ROOT" \
    --summary-root "$OUTPUT_BASE/$ASSET-summary" \
    --run-root "$OUTPUT_BASE" \
    --job-id "$SLURM_JOB_ID" \
    --preflight-job-id "$PREFLIGHT_JOB_ID"
fi
