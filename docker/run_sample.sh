#!/usr/bin/env bash
# ============================================================================
#  Run the built container on the 10 held-out internal-test cases and score it.
#
#  The held-out cases (holdout_test_cases.txt, created by prepare_dataset.py)
#  were NEVER seen during training, and they DO have ground truth, so we can
#  report a real performance number.
#
#  Produces:
#     docker/sample_run/input/    (raw, non-stripped 4-modality cases)
#     docker/sample_run/output/   (container predictions)
#     docker/sample_run/gt/       (ground-truth label maps)
#     docker/sample_run/results.csv (per-case + mean DSC/NSD)
#
#  Usage:  bash docker/run_sample.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
VENV="${PROJECT_ROOT}/venv"
RAW_TRAIN="${PROJECT_ROOT}/data/BraTS26_PED_training_Complete"
HOLDOUT="${PROJECT_ROOT}/holdout_test_cases.txt"
RUN_DIR="${PROJECT_ROOT}/docker/sample_run"
IN="${RUN_DIR}/input"; OUT="${RUN_DIR}/output"; GT="${RUN_DIR}/gt"

[[ -f "${HOLDOUT}" ]] || { echo "[ERROR] ${HOLDOUT} not found — run prepare_dataset.py first."; exit 1; }

rm -rf "${RUN_DIR}"; mkdir -p "${IN}" "${OUT}" "${GT}"

echo "[INFO] Assembling sample input from held-out cases:"
while read -r CID; do
    [[ -z "${CID}" ]] && continue
    mkdir -p "${IN}/${CID}"
    for m in t1n t1c t2w t2f; do
        cp "${RAW_TRAIN}/${CID}/${CID}-${m}.nii.gz" "${IN}/${CID}/"
    done
    cp "${RAW_TRAIN}/${CID}/${CID}-seg.nii.gz" "${GT}/${CID}.nii.gz"
    echo "   + ${CID}"
done < "${HOLDOUT}"

echo "[INFO] Running container..."
if command -v docker >/dev/null 2>&1 && docker image inspect bratsped2026:latest >/dev/null 2>&1; then
    docker run --gpus all --rm -v "${IN}:/input:ro" -v "${OUT}:/output" bratsped2026:latest
elif command -v apptainer >/dev/null 2>&1 && [[ -f "${PROJECT_ROOT}/docker/bratsped2026.sif" ]]; then
    apptainer run --nv --bind "${IN}:/input" --bind "${OUT}:/output" \
        "${PROJECT_ROOT}/docker/bratsped2026.sif"
else
    echo "[ERROR] No built container found (docker image or .sif). Run docker/build.sh first."
    exit 1
fi

echo "[INFO] Scoring predictions against ground truth..."
source "${VENV}/bin/activate"
python "${PROJECT_ROOT}/code/evaluate.py" --pred "${OUT}" --gt "${GT}" \
    --output "${RUN_DIR}/results.csv"

echo "[DONE] Results: ${RUN_DIR}/results.csv"
