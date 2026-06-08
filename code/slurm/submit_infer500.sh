#!/usr/bin/env bash
# ============================================================================
#  Chain 500-epoch ensemble inference AFTER the extend jobs finish.
#
#  Submits (both depend afterok on ALL five extend jobs):
#    - infer.sbatch      → 10 held-out internal-test cases + eval  (TRAINER=500)
#    - val_infer.sbatch  → 91 validation cases                     (TRAINER=500)
#
#  Outputs are namespaced to predictions_500/ so the 250-epoch results
#  (predictions/) are kept for comparison. Uses all 5 folds.
#
#  Usage:
#     # auto-detect the running extend jobs and chain on them:
#     bash code/slurm/submit_infer500.sh
#     # or pass the extend job ids explicitly:
#     bash code/slurm/submit_infer500.sh 510303 510304 510305 510306 510307
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
SLURM_DIR="${PROJECT_ROOT}/code/slurm"
LOG_DIR="${PROJECT_ROOT}/slurm_logs"

TRAINER="${TRAINER:-nnUNetTrainer_500epochs}"
FOLDS_STR="${FOLDS:-0 1 2 3 4}"
PARTS="${PARTS:-rp6b-1-gm96-c8-m64,b200-8-gm1432-c192-m2048}"
ACCOUNT="${SLURM_ACCOUNT:-ai-gpu}"

# ── Resolve the extend job ids to depend on ─────────────────────────────────
DEPS=("$@")
if [[ ${#DEPS[@]} -eq 0 ]]; then
    mapfile -t DEPS < <(squeue -u "$USER" -h -o "%i %j" | awk '$2=="bratsped_extend"{print $1}')
fi
if [[ ${#DEPS[@]} -eq 0 ]]; then
    echo "[FATAL] No extend jobs found to depend on. Pass job ids explicitly." >&2
    exit 1
fi
DEP_SPEC="afterok:$(IFS=:; echo "${DEPS[*]}")"

echo "============================================================"
echo " Chain 500-epoch inference"
echo "   trainer=${TRAINER}  folds=${FOLDS_STR}"
echo "   depends on extend jobs: ${DEPS[*]}"
echo "   partitions=${PARTS}"
echo "============================================================"

mkdir -p "${LOG_DIR}"; cd "${LOG_DIR}"

# ── 1. internal-test inference + evaluation ─────────────────────────────────
IID=$(sbatch --parsable \
    --partition="${PARTS}" --account="${ACCOUNT}" --gpus=1 \
    --dependency="${DEP_SPEC}" \
    --export=ALL,TRAINER="${TRAINER}",FOLDS="${FOLDS_STR}",OUT_DIR="${PROJECT_ROOT}/predictions_500" \
    "${SLURM_DIR}/infer.sbatch")
echo "[SUBMIT] infer500 (internal-test + eval): ${IID}"

# ── 2. validation-set inference ─────────────────────────────────────────────
VID=$(sbatch --parsable \
    --partition="${PARTS}" --account="${ACCOUNT}" --gpus=1 \
    --dependency="${DEP_SPEC}" \
    --export=ALL,TRAINER="${TRAINER}",FOLDS="${FOLDS_STR}",OUT_DIR="${PROJECT_ROOT}/predictions_500/validation_ensemble" \
    "${SLURM_DIR}/val_infer.sbatch")
echo "[SUBMIT] valinfer500 (91 validation cases): ${VID}"

echo "------------------------------------------------------------"
echo "Both fire after: ${DEP_SPEC}"
echo "Results → ${PROJECT_ROOT}/predictions_500/"
