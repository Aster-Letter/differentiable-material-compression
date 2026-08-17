#!/usr/bin/env bash
#SBATCH --job-name=cgf-fish-random-preflight-10
#SBATCH --partition=Students
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=preflight.%j.out
#SBATCH --error=preflight.%j.err

set -euo pipefail

PROJECT_ROOT="${CG_FRONTIER_ROOT:-$HOME/projects/cg_frontier}"
ENV_ROOT="$PROJECT_ROOT/.envs/cg-frontier-system-py312"
BUILD_TMP="$PROJECT_ROOT/.scratch/nvdiffrast-build-$SLURM_JOB_ID"
PYTHON_INCLUDE="$PROJECT_ROOT/.deps/python3.12/include"
NVDIFFRAST_BUILD_SOURCE="$BUILD_TMP/nvdiffrast"

cd "$PROJECT_ROOT"
mkdir -p "$BUILD_TMP"
export TMPDIR="$BUILD_TMP"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
module load cuda/13.0
cp -a "$PROJECT_ROOT/nvdiffrast-253ac4f-source/nvdiffrast" \
  "$NVDIFFRAST_BUILD_SOURCE"
"$ENV_ROOT/bin/python" -m pip install \
  --force-reinstall \
  --no-deps \
  --no-build-isolation \
  "$NVDIFFRAST_BUILD_SOURCE"
MAX_STEPS=10 bash scripts/remote_run_barramundi_random_preflight.sh \
  "$PROJECT_ROOT" \
  "$ENV_ROOT"
