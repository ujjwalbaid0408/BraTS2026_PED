#!/usr/bin/env bash
# ============================================================================
#  BraTS-PED 2026 — EXTEND a finished/partial run to more epochs (SLURM)
#
#  Continues an existing nnU-Net run trained with SRC_TRAINER (e.g.
#  nnUNetTrainer_250epochs) into a longer DST_TRAINER (e.g.
#  nnUNetTrainer_500epochs) WITHOUT retraining from scratch.
#
#  Mechanism (nnU-Net keys the results folder by trainer name, so the longer
#  trainer writes to a *different* folder):
#    1. Seed DST fold dir with the SRC checkpoint (as checkpoint_final.pth) +
#       base plans.json/dataset.json/fingerprint — ONLY if DST has no
#       checkpoint yet (idempotent across requeues).
#    2. Run nnUNetv2_train with the longer trainer and --c. maybe_load_checkpoint
#       loads checkpoint_final.pth, restores current_epoch (~250), and
#       run_training does range(current_epoch, 500) → trains 250..499.
#
#  LR note: the poly LR is recomputed from current_epoch/num_epochs each epoch
#  (it is NOT restored from the checkpoint). A run that finished the 250 decay
#  resumes as a WARM RESTART — LR snaps from ~0 back to ~0.5*initial at the
#  switch, then decays again to 500. This is intended and applied uniformly to
#  every fold for ensemble consistency.
#
#  Submit (see submit_extend.sh):
#     sbatch --export=ALL,FOLD=0 code/slurm/train_extend.sbatch
# ============================================================================
#SBATCH --job-name=bratsped_extend
#SBATCH --partition=b200-8-gm1432-c192-m2048
#SBATCH --account=ai-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ubaid/BraTS2026_PED/slurm_logs/extend_%j.out
#SBATCH --error=/scratch/ubaid/BraTS2026_PED/slurm_logs/extend_%j.err

set -euo pipefail

# ── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
CODE_DIR="${PROJECT_ROOT}/code"
VENV="${PROJECT_ROOT}/venv"

# ── nnU-Net workspace ───────────────────────────────────────────────────────
export nnUNet_raw="${PROJECT_ROOT}/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet/nnUNet_results"
# DA workers = cpus minus a few cores reserved for the main/GPU-feed thread.
# (Training here is CPU-bound on Blosc2 decompress + augmentation; the B200/RTX
#  GPUs idle <15% when starved, so feeding them more DA workers is the win.)
_cpt="${SLURM_CPUS_PER_TASK:-8}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-$(( _cpt > 8 ? _cpt - 4 : _cpt ))}"
export PYTHONUNBUFFERED=1

# ── Config (override via --export) ──────────────────────────────────────────
FOLD="${FOLD:-0}"
SRC_TRAINER="${SRC_TRAINER:-nnUNetTrainer_250epochs}"
DST_TRAINER="${DST_TRAINER:-nnUNetTrainer_500epochs}"
PLANNER="${PLANNER:-nnUNetResEncUNetLPlans}"
DATASET_ID="${DATASET_ID:-21}"
CONFIG="${CONFIG:-3d_fullres}"

echo "=========================================================="
echo " BraTS-PED EXTEND  ${SRC_TRAINER} -> ${DST_TRAINER}"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-NA}"
echo " dataset=${DATASET_ID} config=${CONFIG} fold=${FOLD} planner=${PLANNER}"
echo " GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=========================================================="

source "${VENV}/bin/activate"

# ── Locate SRC + DST folders ────────────────────────────────────────────────
SRC_FOLD=$(find "${nnUNet_results}" -type d \
    -path "*${SRC_TRAINER}__${PLANNER}__${CONFIG}/fold_${FOLD}" 2>/dev/null | head -1 || true)
if [[ -z "${SRC_FOLD}" ]]; then
    echo "[FATAL] No source fold dir for ${SRC_TRAINER} fold ${FOLD}" >&2
    exit 1
fi
SRC_BASE=$(dirname "${SRC_FOLD}")
DATASET_DIR=$(dirname "${SRC_BASE}")
DST_BASE="${DATASET_DIR}/${DST_TRAINER}__${PLANNER}__${CONFIG}"
DST_FOLD="${DST_BASE}/fold_${FOLD}"

# ── Seed DST from SRC (only if DST has no checkpoint yet) ────────────────────
if compgen -G "${DST_FOLD}/checkpoint_*.pth" > /dev/null 2>&1; then
    echo "[INFO] DST already has checkpoint(s) — skipping seed (resuming 500-run)"
else
    echo "[INFO] Seeding ${DST_FOLD} from ${SRC_FOLD}"
    mkdir -p "${DST_FOLD}"
    # base-level metadata (nnU-Net regenerates these, but copy to be safe)
    for f in dataset.json plans.json dataset_fingerprint.json; do
        [[ -f "${SRC_BASE}/${f}" ]] && cp -n "${SRC_BASE}/${f}" "${DST_BASE}/${f}"
    done
    # resume point: prefer the source's final, else its latest
    if [[ -f "${SRC_FOLD}/checkpoint_final.pth" ]]; then
        SRC_CKPT="${SRC_FOLD}/checkpoint_final.pth"
    elif [[ -f "${SRC_FOLD}/checkpoint_latest.pth" ]]; then
        SRC_CKPT="${SRC_FOLD}/checkpoint_latest.pth"
    else
        echo "[FATAL] No checkpoint_final/latest in ${SRC_FOLD}" >&2
        exit 1
    fi
    # --c prefers checkpoint_final.pth → seed under that name
    cp "${SRC_CKPT}" "${DST_FOLD}/checkpoint_final.pth"
    [[ -f "${SRC_FOLD}/checkpoint_best.pth" ]] && \
        cp "${SRC_FOLD}/checkpoint_best.pth" "${DST_FOLD}/checkpoint_best.pth"
    echo "[INFO] Seeded resume checkpoint from $(basename "${SRC_CKPT}")"
    python - "${DST_FOLD}/checkpoint_final.pth" <<'PY' || true
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"[INFO] Seed checkpoint current_epoch={ck.get('current_epoch')} "
      f"trainer={ck.get('trainer_name')}")
PY
fi

# ── Continue training to the longer schedule ────────────────────────────────
srun python "${CODE_DIR}/train.py" \
    --step train \
    --dataset-id "${DATASET_ID}" \
    --folds "${FOLD}" \
    --trainer "${DST_TRAINER}" \
    --planner "${PLANNER}" \
    --c

# ── Save training plots (loss + Dice) ───────────────────────────────────────
python "${CODE_DIR}/plot_training.py" \
    --dataset-id "${DATASET_ID}" --trainer "${DST_TRAINER}" \
    --planner "${PLANNER}" --fold "${FOLD}" || true

echo "[DONE] extend fold ${FOLD} -> ${DST_TRAINER} finished on $(hostname)"
