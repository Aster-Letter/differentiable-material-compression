#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm/c4-render-ablation-lantern-40k-v1
KIND="${1:-}"

case "$KIND" in
  preflight)
    (( $# == 1 )) || { echo "Usage: $0 preflight" >&2; exit 3; }
    sbatch scripts/scow_c4_render_ablation_lantern_40k_preflight.slurm
    ;;
  formal)
    (( $# == 2 )) || { echo "Usage: $0 formal <preflight-job-id>" >&2; exit 3; }
    PREFLIGHT_JOB_ID="$2"
    [[ "$PREFLIGHT_JOB_ID" =~ ^[0-9]+$ ]] || { echo "Preflight job id must be numeric" >&2; exit 3; }
    MARKER="outputs/remote/c4-render-ablation-lantern-40k-v1/${PREFLIGHT_JOB_ID}/preflight_verified.json"
    [[ -f "$MARKER" ]] || { echo "Missing verified preflight marker: $MARKER" >&2; exit 4; }
    sbatch scripts/scow_c4_render_ablation_lantern_40k_job.slurm "$PREFLIGHT_JOB_ID"
    ;;
  resume)
    (( $# == 5 )) || { echo "Usage: $0 resume <material_only|material_render> <failed-job-id> <preflight-job-id> <yes>" >&2; exit 3; }
    ARM="$2"; SOURCE_JOB_ID="$3"; PREFLIGHT_JOB_ID="$4"; CONFIRM="$5"
    case "$ARM" in material_only|material_render) ;; *) echo "Unknown arm: $ARM" >&2; exit 3 ;; esac
    [[ "$SOURCE_JOB_ID" =~ ^[0-9]+$ && "$PREFLIGHT_JOB_ID" =~ ^[0-9]+$ ]] || { echo "Job ids must be numeric" >&2; exit 3; }
    [[ "$CONFIRM" == "yes" ]] || { echo "Resume requires final confirmation token: yes" >&2; exit 3; }
    [[ -z "$(squeue -h -j "$SOURCE_JOB_ID" -o '%i')" ]] || { echo "Source job is still queued or running" >&2; exit 4; }
    CHECKPOINT="outputs/remote/c4-render-ablation-lantern-40k-v1/${SOURCE_JOB_ID}/Lantern/${ARM}/checkpoints/step_30000/checkpoint.pt"
    [[ -f "$CHECKPOINT" ]] || { echo "Missing frozen 30k checkpoint: $CHECKPOINT" >&2; exit 4; }
    C4_LANTERN_40K_MODE=resume sbatch --export=ALL,C4_LANTERN_40K_MODE=resume \
      scripts/scow_c4_render_ablation_lantern_40k_job.slurm \
      "$ARM" "$SOURCE_JOB_ID" "$PREFLIGHT_JOB_ID"
    ;;
  *)
    echo "Usage: $0 <preflight|formal|resume> ..." >&2
    exit 3
    ;;
esac
