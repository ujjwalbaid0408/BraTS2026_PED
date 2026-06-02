#!/usr/bin/env bash
# ============================================================================
#  BraTS-PED 2026 — data preparation (SLURM)
#
#  Runs the CPU/GPU prep chain ONCE before training:
#    1. skull-strip train + val cases   (HD-BET on GPU, or SynthStrip)
#    2. convert to nnU-Net format + hold out 10 internal-test cases
#    3. nnU-Net plan & preprocess (ResEnc-L planner)
#
#  Uses the RTX Pro 6000 node so skull-stripping is GPU-accelerated.
#  Submit:  sbatch code/slurm/prep.sbatch
# ============================================================================
#SBATCH --job-name=bratsped_prep
#SBATCH --partition=rp6b-1-gm96-c8-m64
#SBATCH --account=ai-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/ubaid/BraTS2026_PED/slurm_logs/prep_%j.out
#SBATCH --error=/scratch/ubaid/BraTS2026_PED/slurm_logs/prep_%j.err

set -euo pipefail

PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
CODE_DIR="${PROJECT_ROOT}/code"
VENV="${PROJECT_ROOT}/venv"

export BRATS_PED_ROOT="${PROJECT_ROOT}"
export nnUNet_raw="${PROJECT_ROOT}/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet/nnUNet_results"
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

DATASET_ID="${DATASET_ID:-21}"
SS_METHOD="${SS_METHOD:-auto}"

echo "=== node=$(hostname) job=${SLURM_JOB_ID:-NA} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
source "${VENV}/bin/activate"

echo "=== [1/3] Skull-stripping (method=${SS_METHOD}) ==="
python "${CODE_DIR}/skull_strip.py" --split both --method "${SS_METHOD}" --device gpu

echo "=== [2/3] Prepare nnU-Net dataset (hold out 10 test cases) ==="
python "${CODE_DIR}/prepare_dataset.py"

echo "=== [3/3] Plan & preprocess ==="
python "${CODE_DIR}/train.py" --step preprocess --dataset-id "${DATASET_ID}"

echo "[DONE] preparation complete."
