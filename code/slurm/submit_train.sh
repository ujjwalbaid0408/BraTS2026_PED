#!/usr/bin/env bash
# ============================================================================
#  Resource-aware submission for BraTS-PED training.
#
#  Inspects the cluster (sinfo) BEFORE submitting and picks the GPU partition
#  with available capacity, EXCLUDING the H100/A100 partitions (decommissioned
#  2026-06-03). Then submits:
#       prep job  →  one training job per fold (dependency: afterok)
#
#  Usage:
#     bash code/slurm/submit_train.sh                 # fold 0 only (demo)
#     bash code/slurm/submit_train.sh 0 1 2 3 4       # 5-fold ensemble
#     TRAINER=nnUNetTrainer_250epochs bash code/slurm/submit_train.sh 0
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
SLURM_DIR="${PROJECT_ROOT}/code/slurm"
LOG_DIR="${PROJECT_ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

FOLDS=("$@")
[[ ${#FOLDS[@]} -eq 0 ]] && FOLDS=("0")

PREFERRED_PARTITION="rp6b-1-gm96-c8-m64"      # 1x RTX Pro 6000, 96GB
FALLBACK_PARTITION="l4-4-gm96-c48-m192"        # 4x L4 24GB (use 1)

echo "============================================================"
echo " Cluster GPU availability (excluding H100/A100):"
echo "============================================================"
sinfo -o "%20P %10a %.6D %.8t %12G %N" \
    | grep -Ei "gpu|rtxpro|l4|rp6b" \
    | grep -Eiv "h100|a100|b200" || true
echo "------------------------------------------------------------"

# Pick a partition that has idle/mixed nodes; else default to preferred (dynamic
# partitions report 'idle~' and spin up on demand — still fine).
choose_partition() {
    for part in "${PREFERRED_PARTITION}" "${FALLBACK_PARTITION}"; do
        if sinfo -h -p "${part}" -o "%t" 2>/dev/null | grep -Eq "idle|mix|alloc|down~|idle~"; then
            echo "${part}"; return 0
        fi
    done
    echo "${PREFERRED_PARTITION}"
}
PART=$(choose_partition)
ACCOUNT="${SLURM_ACCOUNT:-ai-gpu}"
echo "[INFO] Selected partition: ${PART}  (account=${ACCOUNT}, --gpus=1)"

cd "${LOG_DIR}"

# ── 1. prep job ─────────────────────────────────────────────────────────────
PREP_ID=$(sbatch --parsable \
    --partition="${PART}" --account="${ACCOUNT}" --gpus=1 \
    "${SLURM_DIR}/prep.sbatch")
echo "[SUBMIT] prep job: ${PREP_ID}"

# ── 2. one training job per fold, depends on prep ───────────────────────────
for f in "${FOLDS[@]}"; do
    TID=$(sbatch --parsable \
        --partition="${PART}" --account="${ACCOUNT}" --gpus=1 \
        --dependency="afterok:${PREP_ID}" \
        --export=ALL,FOLD="${f}",TRAINER="${TRAINER:-nnUNetTrainer}",PLANNER="${PLANNER:-nnUNetResEncUNetLPlans}" \
        "${SLURM_DIR}/train.sbatch")
    echo "[SUBMIT] train fold ${f}: ${TID}  (after prep ${PREP_ID})"
done

echo "------------------------------------------------------------"
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       ${LOG_DIR}"
