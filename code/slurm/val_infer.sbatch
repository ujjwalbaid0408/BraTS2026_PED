#!/usr/bin/env bash
# ============================================================================
#  BraTS-PED 2026 — ensemble inference on the 91 VALIDATION cases (no GT).
#  Saves raw ensemble predictions and lesion-wise post-processed predictions
#  to predictions/validation_ensemble/{raw,postprocessed}.
#
#  Default ensemble = the completed folds (0 1 3 4). Override FOLDS to use all
#  five once fold 2 finishes:  sbatch --export=ALL,FOLDS="0 1 2 3 4" ...
# ============================================================================
#SBATCH --job-name=bratsped_valinfer
#SBATCH --partition=rp6b-1-gm96-c8-m64
#SBATCH --account=ai-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ubaid/BraTS2026_PED/slurm_logs/valinfer_%j.out
#SBATCH --error=/scratch/ubaid/BraTS2026_PED/slurm_logs/valinfer_%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/ubaid/BraTS2026_PED"
CODE_DIR="${PROJECT_ROOT}/code"
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"
PLANNER="nnUNetResEncUNetLPlans"
FOLDS="${FOLDS:-0 1 3 4}"

export nnUNet_raw="${PROJECT_ROOT}/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet/nnUNet_results"
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo " BraTS-PED VALIDATION ensemble inference"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-NA}  folds=${FOLDS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=========================================================="

source "${PROJECT_ROOT}/venv/bin/activate"

SS_VAL="${PROJECT_ROOT}/data_ss/validation"
OUT="${OUT_DIR:-${PROJECT_ROOT}/predictions/validation_ensemble}"
RAW="${OUT}/raw"
POST="${OUT}/postprocessed"
mkdir -p "${RAW}" "${POST}"

# ── Ensemble inference over the validation set (already skull-stripped) ──────
python "${CODE_DIR}/inference.py" \
    --input "${SS_VAL}" --output "${RAW}" --raw-input \
    --trainer "${TRAINER}" --planner "${PLANNER}" \
    --folds ${FOLDS}

# ── Lesion-wise post-processing ─────────────────────────────────────────────
python "${CODE_DIR}/postprocess.py" --input "${RAW}" --output "${POST}"

echo "[DONE] validation ensemble -> ${OUT}"
echo "  raw:           $(ls "${RAW}"/*.nii.gz 2>/dev/null | wc -l) cases"
echo "  postprocessed: $(ls "${POST}"/*.nii.gz 2>/dev/null | wc -l) cases"
