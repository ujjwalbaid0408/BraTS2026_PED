#!/usr/bin/env bash
# ============================================================================
#  setup_env.sh — one-time environment setup for BraTS-PED 2026 (Linux / HPC).
#
#  Creates a virtual environment at <PROJECT_ROOT>/venv, installs the correct
#  CUDA PyTorch wheel, then `pip install -e .` for all project dependencies.
#
#  Usage:
#     bash code/setup_env.sh              # CUDA 12.8 (RTX Pro 6000 / Blackwell)
#     bash code/setup_env.sh --cuda 12.4  # other CUDA 12.x GPUs
#     bash code/setup_env.sh --cpu        # CPU-only (testing; training needs GPU)
# ============================================================================
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BRATS_PED_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"
VENV="${PROJECT_ROOT}/venv"
CUDA_VERSION="12.8"
CPU_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        --cpu)  CPU_ONLY=1; shift ;;
        *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== BraTS-PED env setup ==="
echo " project root : ${PROJECT_ROOT}"
echo " venv         : ${VENV}"
[[ $CPU_ONLY -eq 1 ]] && echo " mode: CPU-only" || echo " CUDA: ${CUDA_VERSION}"

PYTHON_EXE=$(command -v python3.11 || command -v python3)
"$PYTHON_EXE" -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install --upgrade pip wheel setuptools

if [[ $CPU_ONLY -eq 1 ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TAG}"
fi

pip install -e "${CODE_DIR}"

# nnU-Net workspace
mkdir -p "${PROJECT_ROOT}/nnUNet/nnUNet_raw" \
         "${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed" \
         "${PROJECT_ROOT}/nnUNet/nnUNet_results" \
         "${PROJECT_ROOT}/slurm_logs"

cat <<EOF

=== Setup complete ===
  Activate:   source ${VENV}/bin/activate
  Export nnU-Net paths (also set automatically by the pipeline scripts):
    export nnUNet_raw=${PROJECT_ROOT}/nnUNet/nnUNet_raw
    export nnUNet_preprocessed=${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed
    export nnUNet_results=${PROJECT_ROOT}/nnUNet/nnUNet_results

  Next:
    bash code/slurm/submit_train.sh 0          # prep + train fold 0 on RTX Pro 6000
EOF
