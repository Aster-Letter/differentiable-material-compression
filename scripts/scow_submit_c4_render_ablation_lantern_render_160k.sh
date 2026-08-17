#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm/c4-render-ablation-lantern-render-160k-v1
KIND="${1:-}"

active_jobs="$(squeue -h -u "$USER" -t PENDING,RUNNING -o '%i|%T|%j')"
if [[ -n "$active_jobs" ]]; then
  printf 'Refusing to submit: an online job already exists.\n%s\n' "$active_jobs" >&2
  exit 2
fi

case "$KIND" in
  preflight)
    (( $# == 1 )) || { echo "Usage: $0 preflight" >&2; exit 3; }
    sbatch scripts/scow_c4_render_ablation_lantern_render_160k_preflight.slurm
    ;;
  formal)
    (( $# == 2 )) || { echo "Usage: $0 formal <preflight-job-id>" >&2; exit 3; }
    PREFLIGHT_JOB_ID="$2"
    [[ "$PREFLIGHT_JOB_ID" =~ ^[0-9]+$ ]] || { echo "Preflight job id must be numeric" >&2; exit 3; }
    MARKER="outputs/remote/c4-render-ablation-lantern-render-160k-v1/${PREFLIGHT_JOB_ID}/preflight_verified.json"
    [[ -f "$MARKER" ]] || { echo "Missing verified preflight marker: $MARKER" >&2; exit 4; }
    sbatch scripts/scow_c4_render_ablation_lantern_render_160k_job.slurm "$PREFLIGHT_JOB_ID"
    ;;
  resume)
    (( $# == 5 )) || { echo "Usage: $0 resume <failed-160k-job-id> <80000|120000> <preflight-job-id> <yes>" >&2; exit 3; }
    SOURCE_JOB_ID="$2"; CHECKPOINT_STEP="$3"; PREFLIGHT_JOB_ID="$4"; CONFIRM="$5"
    [[ "$SOURCE_JOB_ID" =~ ^[0-9]+$ && "$PREFLIGHT_JOB_ID" =~ ^[0-9]+$ ]] || { echo "Job ids must be numeric" >&2; exit 3; }
    case "$CHECKPOINT_STEP" in 80000|120000) ;; *) echo "Resume step must be 80000 or 120000" >&2; exit 3 ;; esac
    [[ "$CONFIRM" == "yes" ]] || { echo "Resume requires final confirmation token: yes" >&2; exit 3; }
    [[ -z "$(squeue -h -j "$SOURCE_JOB_ID" -o '%i')" ]] || { echo "Source job is still queued or running" >&2; exit 4; }
    CHECKPOINT="outputs/remote/c4-render-ablation-lantern-render-160k-v1/${SOURCE_JOB_ID}/Lantern/material_render/checkpoints/step_$(printf '%06d' "$CHECKPOINT_STEP")/checkpoint.pt"
    [[ -f "$CHECKPOINT" ]] || { echo "Missing resume checkpoint: $CHECKPOINT" >&2; exit 4; }
    C4_LANTERN_RENDER_160K_MODE=resume sbatch --export=ALL,C4_LANTERN_RENDER_160K_MODE=resume \
      scripts/scow_c4_render_ablation_lantern_render_160k_job.slurm \
      "$SOURCE_JOB_ID" "$CHECKPOINT_STEP" "$PREFLIGHT_JOB_ID"
    ;;
  *)
    echo "Usage: $0 <preflight|formal|resume> ..." >&2
    exit 3
    ;;
esac
