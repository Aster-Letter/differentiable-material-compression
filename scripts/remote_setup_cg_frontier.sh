#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"
ENV_ROOT="${2:-$PROJECT_ROOT/.envs/cg-frontier-system-py312}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.ustc.edu.cn/pypi/simple}"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  python3 -m venv "$ENV_ROOT"
fi

"$ENV_ROOT/bin/python" -m pip install --upgrade setuptools wheel ninja --index-url "$PYPI_INDEX"
"$ENV_ROOT/bin/python" -m pip install torch==2.12.0 --index-url "$PYPI_INDEX"
"$ENV_ROOT/bin/python" -m pip install --editable "$PROJECT_ROOT" --index-url "$PYPI_INDEX"

"$ENV_ROOT/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)"
echo "Install nvdiffrast from the pinned local source inside the Slurm GPU allocation."
