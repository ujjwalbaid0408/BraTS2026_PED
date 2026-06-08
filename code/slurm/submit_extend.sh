#!/usr/bin/env bash
# ============================================================================
#  Submit EXTEND jobs (250 -> 500 epochs) for one or more folds.
#
#  For each fold:
#    - If the source run has finished (checkpoint_final.pth present) → submit
#      the extend job immediately.
#    - If the source fold is still training (a running SLURM job writes to it)
#      → submit with --dependency=afterok:<that job> so the extend seeds from
#      the freshly-written checkpoint_final.pth (warm restart at 250, uniform
#      with the already-finished folds).
#
#  Running fold→job mapping is auto-detected by scanning each running job's
#  stdout for its "fold=N" header line.
#
#  Usage:
#     bash code/slurm/submit_extend.sh                 # folds 0..4
#     bash code/slurm/submit_extend.sh 2 4             # specific folds
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
SLURM_DIR="${PROJECT_ROOT}/code/slurm"
LOG_DIR="${PROJECT_ROOT}/slurm_logs"
RES="${PROJECT_ROOT}/nnUNet/nnUNet_results"

SRC_TRAINER="${SRC_TRAINER:-nnUNetTrainer_250epochs}"
DST_TRAINER="${DST_TRAINER:-nnUNetTrainer_500epochs}"
PLANNER="${PLANNER:-nnUNetResEncUNetLPlans}"
CONFIG="${CONFIG:-3d_fullres}"
PART="${PART:-rp6b-1-gm96-c8-m64}"
ACCOUNT="${SLURM_ACCOUNT:-ai-gpu}"

FOLDS=("$@"); [[ ${#FOLDS[@]} -eq 0 ]] && FOLDS=(0 1 2 3 4)
mkdir -p "${LOG_DIR}"; cd "${LOG_DIR}"

# ── Auto-detect running train jobs → which fold each is training ────────────
declare -A FOLD_JOB
while read -r jid st; do
    [[ "${st}" == "R" || "${st}" == "PD" ]] || continue
    out=$(scontrol show job "${jid}" 2>/dev/null | grep -oE "StdOut=\S+" | cut -d= -f2- || true)
    [[ -n "${out}" && -f "${out}" ]] || continue
    fline=$(grep -oE "fold=[0-9]+" "${out}" 2>/dev/null | head -1 || true)
    [[ -n "${fline}" ]] || continue
    f="${fline#fold=}"
    # only care about source-trainer jobs
    if grep -q "trainer=${SRC_TRAINER}" "${out}" 2>/dev/null; then
        FOLD_JOB[$f]="${jid}"
        echo "[DETECT] fold ${f} is being trained by job ${jid} (${st})"
    fi
done < <(squeue -u "$USER" -h -o "%i %t")

echo "============================================================"
echo " Extend ${SRC_TRAINER} -> ${DST_TRAINER}  folds: ${FOLDS[*]}"
echo " partition=${PART} account=${ACCOUNT}"
echo "============================================================"

for f in "${FOLDS[@]}"; do
    src_final="${RES}/Dataset*/${SRC_TRAINER}__${PLANNER}__${CONFIG}/fold_${f}/checkpoint_final.pth"
    dep_args=()
    if compgen -G "${src_final}" > /dev/null 2>&1; then
        note="src finished (final present) → submit now"
    elif [[ -n "${FOLD_JOB[$f]:-}" ]]; then
        dep_args=(--dependency="afterok:${FOLD_JOB[$f]}")
        note="src still training (job ${FOLD_JOB[$f]}) → afterok dependency"
    else
        echo "[SKIP] fold ${f}: no checkpoint_final and no running job found — refusing to seed from a partial mid-training checkpoint. Re-run once it finishes."
        continue
    fi

    JID=$(sbatch --parsable \
        --partition="${PART}" --account="${ACCOUNT}" --gpus=1 \
        "${dep_args[@]}" \
        --export=ALL,FOLD="${f}",SRC_TRAINER="${SRC_TRAINER}",DST_TRAINER="${DST_TRAINER}",PLANNER="${PLANNER}",CONFIG="${CONFIG}" \
        "${SLURM_DIR}/train_extend.sbatch")
    echo "[SUBMIT] extend fold ${f}: ${JID}  (${note})"
done

echo "------------------------------------------------------------"
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       ${LOG_DIR}/extend_*.out"
