#!/usr/bin/env bash
#SBATCH --job-name=cgf-fish-random-80k
#SBATCH --partition=Students
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/random-80k.%j.out
#SBATCH --error=logs/slurm/random-80k.%j.err

set -euo pipefail

PROJECT_ROOT="${CG_FRONTIER_ROOT:-$HOME/projects/cg_frontier}"
ENV_ROOT="$PROJECT_ROOT/.envs/cg-frontier-system-py312"
BUILD_TMP="$PROJECT_ROOT/.scratch/nvdiffrast-build-$SLURM_JOB_ID"
PYTHON_INCLUDE="$PROJECT_ROOT/.deps/python3.12/include"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/remote/barramundi-random-80k-job-$SLURM_JOB_ID"

cleanup() {
  rm -rf -- "$BUILD_TMP"
}
trap cleanup EXIT

cd "$PROJECT_ROOT"
mkdir -p "$BUILD_TMP"
export TMPDIR="$BUILD_TMP"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
module load cuda/13.0

if ! "$ENV_ROOT/bin/python" -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  "$ENV_ROOT/bin/python" -m pip install --no-build-isolation \
    "$PROJECT_ROOT/nvdiffrast-253ac4f-source/nvdiffrast"
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV_ROOT/bin/python" -c "import nvdiffrast.torch, torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))"
"$ENV_ROOT/bin/python" scripts/train_barramundi_c4_render_appearance_5k.py \
  --config configs/train/barramundi_c4_render_random_80k_v1.yaml \
  --candidate F0_random_appearance \
  --output-root "$OUTPUT_ROOT"

"$ENV_ROOT/bin/python" -c "import hashlib,json,pathlib; root=pathlib.Path('$OUTPUT_ROOT'); d=json.loads((root/'summary.json').read_text()); t=d['candidates'][0]['endpoint']; r=json.loads((root/'F0_random_appearance/training_report.json').read_text()); assert d['status']=='complete_bounded_run'; assert d['initialization']['pca_used'] is False; assert d['runtime_contract']['metallic_affine_row_frozen'] is False; assert [x['step'] for x in r['trajectory']]==[10000,20000,40000,60000,80000]; assert t['step']==80000; assert all(hashlib.sha256((root/x['checkpoint']).read_bytes()).hexdigest()==x['checkpoint_sha256'] for x in r['trajectory']); print('random_80k_summary_ok', root/'summary.json')"
