#!/usr/bin/env bash
# ============================================================================
#  Stage the trained model into docker/model/ and build the container.
#
#  Auto-detects Docker or Apptainer. Run from anywhere; paths are absolute.
#
#  Usage:
#     bash docker/build.sh                       # final checkpoint, fold 0
#     CHECKPOINT=checkpoint_best.pth FOLDS="0" bash docker/build.sh
#     FOLDS="0 1 2 3 4" bash docker/build.sh     # bake a 5-fold ensemble
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${BRATS_PED_ROOT:-/scratch/ubaid/BraTS2026_PED}"
DOCKER_DIR="${PROJECT_ROOT}/docker"
RESULTS="${nnUNet_results:-${PROJECT_ROOT}/nnUNet/nnUNet_results}"

DATASET_DIR_NAME="Dataset021_BraTSPED2026"
TRAINER="${TRAINER:-nnUNetTrainer}"
PLANNER="${PLANNER:-nnUNetResEncUNetLPlans}"
CONFIG="${CONFIG:-3d_fullres}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
FOLDS="${FOLDS:-0}"

SRC="${RESULTS}/${DATASET_DIR_NAME}/${TRAINER}__${PLANNER}__${CONFIG}"
DST="${DOCKER_DIR}/model/${DATASET_DIR_NAME}/${TRAINER}__${PLANNER}__${CONFIG}"

if [[ ! -d "${SRC}" ]]; then
    echo "[ERROR] Trained model not found at: ${SRC}"
    echo "        Train first (code/slurm/submit_train.sh) before building."
    exit 1
fi

echo "[INFO] Staging slim model: ${SRC} -> ${DST}"
rm -rf "${DOCKER_DIR}/model"
mkdir -p "${DST}"
# inference-time essentials
for f in plans.json dataset.json dataset_fingerprint.json; do
    [[ -f "${SRC}/${f}" ]] && cp "${SRC}/${f}" "${DST}/"
done
for fold in ${FOLDS}; do
    mkdir -p "${DST}/fold_${fold}"
    cp "${SRC}/fold_${fold}/${CHECKPOINT}" "${DST}/fold_${fold}/checkpoint_final.pth"
done
echo "[INFO] Staged folds: ${FOLDS} (checkpoint=${CHECKPOINT})"
du -sh "${DOCKER_DIR}/model" || true

cd "${PROJECT_ROOT}"
if command -v docker >/dev/null 2>&1; then
    echo "[INFO] Building Docker image bratsped2026:latest"
    docker build -f docker/Dockerfile -t bratsped2026:latest .
elif command -v apptainer >/dev/null 2>&1; then
    echo "[INFO] Building Apptainer image docker/bratsped2026.sif"
    apptainer build "${DOCKER_DIR}/bratsped2026.sif" "${DOCKER_DIR}/apptainer.def"
else
    echo "[ERROR] Neither docker nor apptainer found."
    exit 1
fi
echo "[DONE] Container built. See docker/run_sample.sh to run on sample cases."
