#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${PPMI_PYTHON_ENV:?Set PPMI_PYTHON_ENV to the Python environment directory}"
source "$PPMI_PYTHON_ENV/bin/activate"
python work/ppmi-qc-sensitivity-2026-09-12/finalize_gpu.py --gpu-check > work/ppmi-qc-sensitivity-2026-09-12/gpu_preflight.log 2>&1
Rscript work/ppmi-qc-sensitivity-2026-09-12/run_sensitivity.R > work/ppmi-qc-sensitivity-2026-09-12/run.log 2>&1
python work/ppmi-qc-sensitivity-2026-09-12/finalize_gpu.py > work/ppmi-qc-sensitivity-2026-09-12/finalize.log 2>&1
