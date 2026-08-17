#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"
ENV_ROOT="${2:-$PROJECT_ROOT/.envs/cg-frontier-system-py312}"
MAX_STEPS="${MAX_STEPS:-10}"
HOST_NAME="$(hostname)"
NVDIFFRAST_SOURCE="${NVDIFFRAST_SOURCE:-$PROJECT_ROOT/nvdiffrast-253ac4f-source/nvdiffrast}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to train outside a Slurm allocation." >&2
  exit 2
fi
case "$HOST_NAME" in
  *tradmin*|*login*)
    echo "Refusing to train on login node: $HOST_NAME" >&2
    exit 2
    ;;
esac
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  echo "Missing environment Python: $ENV_ROOT/bin/python" >&2
  exit 3
fi

module load cuda/13.0
cd "$PROJECT_ROOT"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV_ROOT/bin/python" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))"
if ! "$ENV_ROOT/bin/python" -c "import nvdiffrast.torch" >/dev/null 2>&1; then
  "$ENV_ROOT/bin/python" -m pip install --no-build-isolation "$NVDIFFRAST_SOURCE"
fi
"$ENV_ROOT/bin/python" -c "import nvdiffrast.torch; print('nvdiffrast_import', True)"

OUTPUT_ROOT="outputs/remote/barramundi-random-preflight-job-${SLURM_JOB_ID}"
"$ENV_ROOT/bin/python" scripts/train_barramundi_c4_render_appearance_5k.py \
  --config configs/train/barramundi_c4_render_random_5k_v1.yaml \
  --max-steps "$MAX_STEPS" \
  --candidate F0_random_appearance \
  --output-root "$OUTPUT_ROOT"

"$ENV_ROOT/bin/python" -c "import json, math, pathlib; p=pathlib.Path('$OUTPUT_ROOT/summary.json'); d=json.loads(p.read_text()); assert d['status']=='complete_bounded_run'; assert d['initialization']['pca_used'] is False; assert d['runtime_contract']['metallic_affine_row_frozen'] is False; assert all(math.isfinite(float(v)) for v in d['parent']['material'].values() if isinstance(v, (int, float))); print('preflight_summary_ok', p)"
