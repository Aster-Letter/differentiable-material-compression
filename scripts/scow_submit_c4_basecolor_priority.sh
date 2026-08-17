#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/cg_frontier}"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm/c4-basecolor-priority

active_jobs="$(squeue -h -u "$USER" -t PENDING,RUNNING -o '%i|%T|%j')"
if [[ -n "$active_jobs" ]]; then
  printf 'Refusing to submit: an online job already exists.\n%s\n' "$active_jobs" >&2
  exit 2
fi
if [[ $# -lt 2 ]]; then
  echo "Usage: scow_submit_c4_basecolor_priority.sh <preflight|formal> <mode> [mode args]" >&2
  exit 3
fi

KIND="$1"
shift
case "$KIND" in
  preflight) SCRIPT="scripts/scow_c4_basecolor_priority_preflight.slurm" ;;
  formal) SCRIPT="scripts/scow_c4_basecolor_priority_job.slurm" ;;
  *) echo "Unknown submission kind: $KIND" >&2; exit 3 ;;
esac

sbatch "$SCRIPT" "$@"
