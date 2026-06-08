#!/usr/bin/env bash
# ============================================================================
#  BraTS-PED 2026 — nnU-Net training (SLURM)
#
#  Target partition: rp6b-1-gm96-c8-m64  (1x NVIDIA RTX Pro 6000, 96 GB VRAM,
#  8 CPU cores, 64 GB RAM, dynamic nodes — chosen to AVOID the H100/A100
#  partitions that are decommissioned on 2026-06-03).
#
#  Auto-resume: nnU-Net writes checkpoint_latest.pth every 50 epochs and
#  checkpoint_best.pth on best EMA Dice. If this job is killed by a time/GPU
#  limit it is automatically requeued (--requeue); on restart we detect the
#  existing checkpoint and continue with `--c`. So a GPU-timeout never loses
#  more than ~50 epochs of progress.
#
#  Submit (see submit_train.sh for resource checks):
#     sbatch --export=ALL,FOLD=0 code/slurm/train.sbatch
#  Override trainer/planner:
#     sbatch --export=ALL,FOLD=0,TRAINER=nnUNetTrainer_250epochs code/slurm/train.sbatch
# ============================================================================
#SBATCH --job-name=bratsped_train
#SBATCH --partition=rp6b-1-gm96-c8-m64
#SBATCH --account=ai-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ubaid/BraTS2026_PED/slurm_logs/train_%j.out
#SBATCH --error=/scratch/ubaid/BraTS2026_PED/slurm_logs/train_%j.err

set -euo pipefail

# ── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
CODE_DIR="${PROJECT_ROOT}/code"
VENV="${PROJECT_ROOT}/venv"

# ── nnU-Net workspace ───────────────────────────────────────────────────────
export nnUNet_raw="${PROJECT_ROOT}/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet/nnUNet_results"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-${SLURM_CPUS_PER_TASK:-8}}"
export PYTHONUNBUFFERED=1

# ── Training config (override via --export) ─────────────────────────────────
FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"
PLANNER="${PLANNER:-nnUNetResEncUNetLPlans}"
DATASET_ID="${DATASET_ID:-21}"
CONFIG="${CONFIG:-3d_fullres}"

echo "=========================================================="
echo " BraTS-PED training"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-NA}"
echo " dataset=${DATASET_ID} config=${CONFIG} fold=${FOLD}"
echo " trainer=${TRAINER} planner=${PLANNER}"
echo " GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=========================================================="

source "${VENV}/bin/activate"

# ── Detect existing checkpoint → resume ─────────────────────────────────────
FOLD_DIR=$(find "${nnUNet_results}" -type d -path "*${TRAINER}__${PLANNER}__${CONFIG}/fold_${FOLD}" 2>/dev/null | head -1 || true)
RESUME_FLAG=""
if [[ -n "${FOLD_DIR}" ]] && compgen -G "${FOLD_DIR}/checkpoint_*.pth" > /dev/null 2>&1; then
    echo "[INFO] Found checkpoint(s) in ${FOLD_DIR} — resuming with --c"
    RESUME_FLAG="--c"
fi

# ── Train (use srun so SLURM can signal/requeue cleanly) ────────────────────
srun python "${CODE_DIR}/train.py" \
    --step train \
    --dataset-id "${DATASET_ID}" \
    --folds "${FOLD}" \
    --trainer "${TRAINER}" \
    --planner "${PLANNER}" \
    ${RESUME_FLAG}

# ── Save training plots (loss + Dice) ───────────────────────────────────────
python "${CODE_DIR}/plot_training.py" \
    --dataset-id "${DATASET_ID}" --trainer "${TRAINER}" \
    --planner "${PLANNER}" --fold "${FOLD}" || true

echo "[DONE] fold ${FOLD} finished on $(hostname)"
