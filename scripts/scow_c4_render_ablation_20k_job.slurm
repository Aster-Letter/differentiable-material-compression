#!/usr/bin/env bash
#SBATCH --job-name=c4-render-ablation-20k
#SBATCH --partition=Students
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/c4-render-ablation-20k-v1/%x.%j.out
#SBATCH --error=logs/slurm/c4-render-ablation-20k-v1/%x.%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "$PROJECT_ROOT"
exec bash scripts/remote_run_c4_render_ablation_20k.sh "${C4_ABLATION_MODE:-formal}" "$@"
