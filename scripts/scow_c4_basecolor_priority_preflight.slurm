#!/usr/bin/env bash
#SBATCH --job-name=c4-bc-preflight
#SBATCH --partition=Students
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/c4-basecolor-priority/%x.%j.out
#SBATCH --error=logs/slurm/c4-basecolor-priority/%x.%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/cg_frontier}"
cd "$PROJECT_ROOT"
MAX_STEPS="${MAX_STEPS:-10}"
export MAX_STEPS
exec bash scripts/remote_run_c4_basecolor_priority.sh "$@"
