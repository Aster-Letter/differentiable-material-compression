#!/usr/bin/env bash
#SBATCH --job-name=cgf-fish-random-5k
#SBATCH --partition=Students
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX5090:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=random-5k.%j.out
#SBATCH --error=random-5k.%j.err

set -euo pipefail

PROJECT_ROOT="${CG_FRONTIER_ROOT:-$HOME/projects/cg_frontier}"
ENV_ROOT="$PROJECT_ROOT/.envs/cg-frontier-system-py312"
BUILD_TMP="$PROJECT_ROOT/.scratch/nvdiffrast-build-$SLURM_JOB_ID"
PYTHON_INCLUDE="$PROJECT_ROOT/.deps/python3.12/include"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/remote/barramundi-random-5k-job-$SLURM_JOB_ID"

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
  --config configs/train/barramundi_c4_render_random_5k_v1.yaml \
  --max-steps 5000 \
  --candidate F0_random_appearance \
  --output-root "$OUTPUT_ROOT"

"$ENV_ROOT/bin/python" -c "import hashlib, json, pathlib; root=pathlib.Path('$OUTPUT_ROOT'); d=json.loads((root/'summary.json').read_text()); e=d['candidates'][0]['endpoint']; cp=root/e['checkpoint']; assert d['status']=='complete_bounded_run'; assert e['step']==5000; assert d['initialization']['pca_used'] is False; assert d['runtime_contract']['metallic_affine_row_frozen'] is False; assert hashlib.sha256(cp.read_bytes()).hexdigest()==e['checkpoint_sha256']; print('random_5k_summary_ok', root/'summary.json')"
