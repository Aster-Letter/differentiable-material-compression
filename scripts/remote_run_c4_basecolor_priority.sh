#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/projects/cg_frontier}"
ENV_ROOT="${ENV_ROOT:-$PROJECT_ROOT/.envs/cg-frontier-system-py312}"
MODE="${1:-}"
shift || true
HOST_NAME="$(hostname)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Refusing to run C4 training outside a Slurm allocation." >&2
  exit 2
fi
case "$HOST_NAME" in
  *tradmin*|*login*)
    echo "Refusing to run on login node: $HOST_NAME" >&2
    exit 2
    ;;
esac
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  echo "Missing environment Python: $ENV_ROOT/bin/python" >&2
  exit 3
fi
if [[ -z "$MODE" ]]; then
  echo "Usage: remote_run_c4_basecolor_priority.sh <mode> [mode arguments]" >&2
  exit 4
fi

module load cuda/13.0
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export TMPDIR="$PROJECT_ROOT/.scratch/c4-basecolor-priority-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
OUTPUT_BASE="outputs/remote/c4-basecolor-priority/${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"
STEP_ARGS=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  STEP_ARGS=(--max-steps "$MAX_STEPS")
fi

echo "job_id=$SLURM_JOB_ID"
echo "host=$HOST_NAME"
echo "project_root=$PROJECT_ROOT"
echo "mode=$MODE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV_ROOT/bin/python" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"

case "$MODE" in
  asset-screen)
    "$ENV_ROOT/bin/python" scripts/audit_complex_c4_assets.py \
      --config configs/eval/complex_c4_asset_screen_v1.yaml \
      --output-root "$OUTPUT_BASE/asset-screen" \
      "$@"
    ;;
  helmet-audit)
    "$ENV_ROOT/bin/python" scripts/train_scifihelmet_c4_basecolor_priority_10k.py \
      --config configs/train/scifihelmet_c4_basecolor_priority_10k_v1.yaml \
      --audit-only \
      --output-root "$OUTPUT_BASE/helmet-gradient-audit" \
      "$@"
    ;;
  helmet-candidate)
    CANDIDATE="${1:?helmet-candidate requires candidate id}"
    AUDIT_REPORT="${2:?helmet-candidate requires gradient_audit.json}"
    shift 2
    "$ENV_ROOT/bin/python" scripts/train_scifihelmet_c4_basecolor_priority_10k.py \
      --config configs/train/scifihelmet_c4_basecolor_priority_10k_v1.yaml \
      --candidate "$CANDIDATE" \
      --audit-report "$AUDIT_REPORT" \
      --output-root "$OUTPUT_BASE/helmet-$CANDIDATE" \
      "${STEP_ARGS[@]}" \
      "$@"
    ;;
  helmet-oracle)
    SOURCE_CANDIDATE="${1:?helmet-oracle requires BC80 or BC90}"
    SOURCE_CHECKPOINT="${2:?helmet-oracle requires source checkpoint}"
    AUDIT_REPORT="${3:?helmet-oracle requires gradient_audit.json}"
    shift 3
    "$ENV_ROOT/bin/python" scripts/train_scifihelmet_c4_basecolor_priority_10k.py \
      --config configs/train/scifihelmet_c4_basecolor_priority_10k_v1.yaml \
      --posthoc-oracle-from "$SOURCE_CHECKPOINT" \
      --source-candidate "$SOURCE_CANDIDATE" \
      --audit-report "$AUDIT_REPORT" \
      --output-root "$OUTPUT_BASE/helmet-$SOURCE_CANDIDATE-oracle" \
      --allow-compander \
      "$@"
    ;;
  helmet-summary)
    "$ENV_ROOT/bin/python" scripts/render_scifihelmet_c4_basecolor_priority_summary.py \
      --config configs/train/scifihelmet_c4_basecolor_priority_10k_v1.yaml \
      --output-root "$OUTPUT_BASE/helmet-summary" \
      "$@"
    ;;
  complex-audit)
    ASSET="${1:?complex-audit requires selected asset id}"
    SCREEN_SUMMARY="${2:?complex-audit requires Phase 0 summary}"
    SCREEN_SHA="${3:?complex-audit requires Phase 0 summary SHA-256}"
    shift 3
    "$ENV_ROOT/bin/python" scripts/train_complex_c4_basecolor_priority_10k.py \
      --config configs/train/complex_c4_basecolor_priority_10k_v1.yaml \
      --screen-summary "$SCREEN_SUMMARY" \
      --screen-summary-sha256 "$SCREEN_SHA" \
      --asset "$ASSET" \
      --audit-only \
      --output-root "$OUTPUT_BASE/$ASSET-gradient-audit" \
      --allow-phase3 \
      "$@"
    ;;
  complex-candidate)
    ASSET="${1:?complex-candidate requires selected asset id}"
    CANDIDATE="${2:?complex-candidate requires candidate id}"
    SCREEN_SUMMARY="${3:?complex-candidate requires Phase 0 summary}"
    SCREEN_SHA="${4:?complex-candidate requires Phase 0 summary SHA-256}"
    AUDIT_REPORT="${5:?complex-candidate requires gradient_audit.json}"
    shift 5
    "$ENV_ROOT/bin/python" scripts/train_complex_c4_basecolor_priority_10k.py \
      --config configs/train/complex_c4_basecolor_priority_10k_v1.yaml \
      --screen-summary "$SCREEN_SUMMARY" \
      --screen-summary-sha256 "$SCREEN_SHA" \
      --asset "$ASSET" \
      --candidate "$CANDIDATE" \
      --audit-report "$AUDIT_REPORT" \
      --output-root "$OUTPUT_BASE/$ASSET-$CANDIDATE" \
      --allow-phase3 \
      "${STEP_ARGS[@]}" \
      "$@"
    ;;
  complex-summary)
    ASSET="${1:?complex-summary requires selected asset id}"
    SCREEN_SUMMARY="${2:?complex-summary requires Phase 0 summary}"
    SCREEN_SHA="${3:?complex-summary requires Phase 0 summary SHA-256}"
    shift 3
    "$ENV_ROOT/bin/python" scripts/render_complex_c4_basecolor_priority_summary.py \
      --config configs/train/complex_c4_basecolor_priority_10k_v1.yaml \
      --screen-summary "$SCREEN_SUMMARY" \
      --screen-summary-sha256 "$SCREEN_SHA" \
      --asset "$ASSET" \
      --output-root "$OUTPUT_BASE/$ASSET-summary" \
      "$@"
    ;;
  *)
    echo "Unknown C4 BaseColor mode: $MODE" >&2
    exit 4
    ;;
esac

"$ENV_ROOT/bin/python" - <<'PY'
import os
from pathlib import Path

root = Path("outputs/remote/c4-basecolor-priority") / os.environ["SLURM_JOB_ID"]
reports = sorted(root.rglob("*.json"))
if not reports:
    raise SystemExit("No JSON result was written")
print("result_root", root)
print("json_reports", len(reports))
PY
