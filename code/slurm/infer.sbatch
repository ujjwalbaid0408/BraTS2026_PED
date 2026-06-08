#!/usr/bin/env bash
# ============================================================================
#  BraTS-PED 2026 — 5-fold ensemble inference on the 10 held-out internal-test
#  cases, then lesion-wise post-processing and (best-effort) evaluation.
#
#  Designed to be submitted with a dependency on the fold-training jobs:
#     sbatch --dependency=afterok:<fold1>:<fold2> code/slurm/infer.sbatch
#  so it fires automatically once all 5 folds have checkpoint_final.pth.
# ============================================================================
#SBATCH --job-name=bratsped_infer
#SBATCH --partition=rp6b-1-gm96-c8-m64
#SBATCH --account=ai-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/ubaid/BraTS2026_PED/slurm_logs/infer_%j.out
#SBATCH --error=/scratch/ubaid/BraTS2026_PED/slurm_logs/infer_%j.err

set -euo pipefail

PROJECT_ROOT="/scratch/ubaid/BraTS2026_PED"
CODE_DIR="${PROJECT_ROOT}/code"
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"
PLANNER="nnUNetResEncUNetLPlans"
# Output base — override (e.g. predictions_500) to keep 250 vs 500 side by side
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/predictions}"
mkdir -p "${OUT_DIR}"

export nnUNet_raw="${PROJECT_ROOT}/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet/nnUNet_results"
export PYTHONUNBUFFERED=1

echo "=========================================================="
echo " BraTS-PED ensemble inference"
echo " node=$(hostname)  job=${SLURM_JOB_ID:-NA}"
echo " trainer=${TRAINER} planner=${PLANNER} folds=0 1 2 3 4"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=========================================================="

source "${PROJECT_ROOT}/venv/bin/activate"

# ── 1. Build held-out internal-test input dir (skull-stripped cases) ────────
TEST_IN="${PROJECT_ROOT}/data_ss/internal_test"
rm -rf "${TEST_IN}"; mkdir -p "${TEST_IN}"
while read -r cid; do
    [[ -z "${cid}" ]] && continue
    ln -sfn "${PROJECT_ROOT}/data_ss/training/${cid}" "${TEST_IN}/${cid}"
done < "${PROJECT_ROOT}/holdout_test_cases.txt"
echo "[INFO] internal-test cases: $(ls "${TEST_IN}" | wc -l)"

# ── 2. Ensemble inference (default = completed folds 0 1 3 4) ───────────────
#  Override with: sbatch --export=ALL,FOLDS="0 1 2 3 4" ... once fold 2 lands.
FOLDS="${FOLDS:-0 1 3 4}"
PRED="${OUT_DIR}/raw"
python "${CODE_DIR}/inference.py" \
    --input "${TEST_IN}" --output "${PRED}" --raw-input \
    --trainer "${TRAINER}" --planner "${PLANNER}" \
    --folds ${FOLDS}

# ── 3. Lesion-wise post-processing ─────────────────────────────────────────
POST="${OUT_DIR}/postprocessed"
python "${CODE_DIR}/postprocess.py" --input "${PRED}" --output "${POST}"

# ── 4. Build GT dir from held-out segs and evaluate (best-effort) ──────────
GT="${OUT_DIR}/gt"
rm -rf "${GT}"; mkdir -p "${GT}"
while read -r cid; do
    [[ -z "${cid}" ]] && continue
    cp "${PROJECT_ROOT}/data_ss/training/${cid}/${cid}-seg.nii.gz" "${GT}/${cid}.nii.gz"
done < "${PROJECT_ROOT}/holdout_test_cases.txt"

echo "[INFO] Evaluating raw ensemble..."
python "${CODE_DIR}/evaluate.py" --pred "${PRED}" --gt "${GT}" \
    --output "${OUT_DIR}/results_raw.csv" \
    || echo "[WARN] evaluate (raw) failed — check label scheme (METS vs PED)"
echo "[INFO] Evaluating post-processed ensemble..."
python "${CODE_DIR}/evaluate.py" --pred "${POST}" --gt "${GT}" \
    --output "${OUT_DIR}/results_postproc.csv" \
    || echo "[WARN] evaluate (postproc) failed — check label scheme (METS vs PED)"

# ── 5. Lesion-wise eval (official BraTS metric: LesionWise Dice + HD95) ──────
echo "[INFO] Lesion-wise eval (raw)..."
python "${CODE_DIR}/evaluate_lesionwise.py" --pred "${PRED}" --gt "${GT}" \
    --output "${OUT_DIR}/results_raw_lesionwise.csv" \
    || echo "[WARN] lesion-wise evaluate (raw) failed"
echo "[INFO] Lesion-wise eval (post-processed)..."
python "${CODE_DIR}/evaluate_lesionwise.py" --pred "${POST}" --gt "${GT}" \
    --output "${OUT_DIR}/results_postproc_lesionwise.csv" \
    || echo "[WARN] lesion-wise evaluate (postproc) failed"

echo "[DONE] ensemble inference + post-processing complete on $(hostname)"
