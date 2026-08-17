#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
mkdir -p logs/slurm/c4-render-ablation-20k-v1

active_jobs="$(squeue -h -u "$USER" -t PENDING,RUNNING -o '%i|%T|%j')"
if [[ -n "$active_jobs" ]]; then
  printf 'Refusing to submit: an online job already exists.\n%s\n' "$active_jobs" >&2
  exit 2
fi
KIND="${1:-}"
case "$KIND" in
  preflight)
    if (( $# != 1 )); then echo "Usage: $0 preflight" >&2; exit 3; fi
    sbatch scripts/scow_c4_render_ablation_20k_preflight.slurm
    ;;
  formal)
    if (( $# != 3 )); then echo "Usage: $0 formal <Corset|Lantern|BoomBox> <preflight-job-id>" >&2; exit 3; fi
    ASSET="$2"
    PREFLIGHT_JOB_ID="$3"
    case "$ASSET" in Corset|Lantern|BoomBox) ;; *) echo "Unknown asset: $ASSET" >&2; exit 3 ;; esac
    if ! [[ "$PREFLIGHT_JOB_ID" =~ ^[0-9]+$ ]]; then
      echo "Preflight job id must be numeric" >&2
      exit 3
    fi
    PREFLIGHT_STATE="$(sacct -n -X -j "$PREFLIGHT_JOB_ID" --format=State:20 -P | head -n 1 | cut -d'|' -f1 | tr -d ' ')"
    if [[ "$PREFLIGHT_STATE" != "COMPLETED" ]]; then
      echo "Preflight job $PREFLIGHT_JOB_ID is not COMPLETED: ${PREFLIGHT_STATE:-unknown}" >&2
      exit 4
    fi
    MARKER="outputs/remote/c4-render-ablation-20k-v1/${PREFLIGHT_JOB_ID}/preflight_verified.json"
    if [[ ! -f "$MARKER" ]]; then
      echo "Preflight job lacks verified marker: $MARKER" >&2
      exit 4
    fi
    sbatch scripts/scow_c4_render_ablation_20k_job.slurm "$ASSET" "$PREFLIGHT_JOB_ID"
    ;;
  resume)
    if (( $# != 5 )); then echo "Usage: $0 resume <asset> <arm> <source-job-id> <preflight-job-id>" >&2; exit 3; fi
    ASSET="$2"; ARM="$3"; SOURCE_JOB_ID="$4"; PREFLIGHT_JOB_ID="$5"
    case "$ASSET" in Corset|Lantern|BoomBox) ;; *) echo "Unknown asset: $ASSET" >&2; exit 3 ;; esac
    case "$ARM" in material_only|material_render) ;; *) echo "Unknown arm: $ARM" >&2; exit 3 ;; esac
    for value in "$SOURCE_JOB_ID" "$PREFLIGHT_JOB_ID"; do
      [[ "$value" =~ ^[0-9]+$ ]] || { echo "Job ids must be numeric" >&2; exit 3; }
    done
    SOURCE_STATE="$(sacct -n -X -j "$SOURCE_JOB_ID" --format=State:20 -P | head -n 1 | cut -d'|' -f1 | tr -d ' ')"
    if [[ "$SOURCE_STATE" != "FAILED" && "$SOURCE_STATE" != "TIMEOUT" && "$SOURCE_STATE" != "OUT_OF_MEMORY" ]]; then
      echo "Resume source job is not in a failed terminal state: ${SOURCE_STATE:-unknown}" >&2
      exit 4
    fi
    PREFLIGHT_STATE="$(sacct -n -X -j "$PREFLIGHT_JOB_ID" --format=State:20 -P | head -n 1 | cut -d'|' -f1 | tr -d ' ')"
    MARKER="outputs/remote/c4-render-ablation-20k-v1/${PREFLIGHT_JOB_ID}/preflight_verified.json"
    if [[ "$PREFLIGHT_STATE" != "COMPLETED" || ! -f "$MARKER" ]]; then
      echo "Resume preflight is not both COMPLETED and verified: $PREFLIGHT_JOB_ID" >&2
      exit 4
    fi
    C4_ABLATION_MODE=resume sbatch --export=ALL,C4_ABLATION_MODE=resume \
      scripts/scow_c4_render_ablation_20k_job.slurm "$ASSET" "$ARM" "$SOURCE_JOB_ID" "$PREFLIGHT_JOB_ID"
    ;;
  *) echo "Usage: $0 <preflight|formal|resume> ..." >&2; exit 3 ;;
esac
