#!/usr/bin/env bash
# ============================================================================
#  run_pipeline.sh — local (non-SLURM) BraTS-PED pipeline driver.
#
#  Steps:
#    strip     skull-strip train + val cases
#    prepare   convert skull-stripped data -> nnU-Net format (+ hold out 10)
#    preprocess nnU-Net plan & preprocess
#    train     train (default fold 0; pass extra flags e.g. --folds 0 1 2 3 4)
#    infer     predict held-out internal-test cases
#    postproc  lesion-wise post-processing
#    evaluate  DSC / NSD vs ground truth
#    all       strip -> prepare -> preprocess -> train -> infer -> postproc -> evaluate
#
#  For long training runs on the cluster use code/slurm/submit_train.sh instead.
#  Run setup_env.sh once before using this script.
# ============================================================================
set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BRATS_PED_ROOT:-$(cd "${CODE_DIR}/.." && pwd)}"
VENV="${PROJECT_ROOT}/venv"

[[ -f "${VENV}/bin/activate" ]] || { echo "[ERROR] venv missing at ${VENV}. Run bash code/setup_env.sh"; exit 1; }
source "${VENV}/bin/activate"

export BRATS_PED_ROOT="${PROJECT_ROOT}"
export nnUNet_raw="${PROJECT_ROOT}/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet/nnUNet_results"
export PYTHONUNBUFFERED=1

DATASET_DIR="${nnUNet_raw}/Dataset021_BraTSPED2026"
PRED_DIR="${PROJECT_ROOT}/predictions/raw"
POSTPROC_DIR="${PROJECT_ROOT}/predictions/postprocessed"
GT_DIR="${DATASET_DIR}/labelsTr"
RESULTS_CSV="${PROJECT_ROOT}/predictions/results.csv"
SS_VAL="${PROJECT_ROOT}/data_ss/validation"

STEP="${1:-all}"; shift || true
EXTRA="$*"

echo "==== BraTS-PED pipeline — step: ${STEP} ===="

if [[ "$STEP" == "strip" || "$STEP" == "all" ]]; then
    python "${CODE_DIR}/skull_strip.py" --split both --method auto --device gpu
    [[ "$STEP" == "strip" ]] && exit 0
fi
if [[ "$STEP" == "prepare" || "$STEP" == "all" ]]; then
    python "${CODE_DIR}/prepare_dataset.py"
    [[ "$STEP" == "prepare" ]] && exit 0
fi
if [[ "$STEP" == "preprocess" || "$STEP" == "all" ]]; then
    python "${CODE_DIR}/train.py" --step preprocess
    [[ "$STEP" == "preprocess" ]] && exit 0
fi
if [[ "$STEP" == "train" || "$STEP" == "all" ]]; then
    # shellcheck disable=SC2086
    python "${CODE_DIR}/train.py" --step train ${EXTRA:---folds 0}
    python "${CODE_DIR}/plot_training.py" || true
    [[ "$STEP" == "train" ]] && exit 0
fi
if [[ "$STEP" == "infer" || "$STEP" == "all" ]]; then
    mkdir -p "${PRED_DIR}"
    # shellcheck disable=SC2086
    python "${CODE_DIR}/inference.py" --input "${SS_VAL}" --output "${PRED_DIR}" --raw-input ${EXTRA}
    [[ "$STEP" == "infer" ]] && exit 0
fi
if [[ "$STEP" == "postproc" || "$STEP" == "all" ]]; then
    mkdir -p "${POSTPROC_DIR}"
    python "${CODE_DIR}/postprocess.py" --input "${PRED_DIR}" --output "${POSTPROC_DIR}"
    [[ "$STEP" == "postproc" ]] && exit 0
fi
if [[ "$STEP" == "evaluate" || "$STEP" == "all" ]]; then
    python "${CODE_DIR}/evaluate.py" --pred "${POSTPROC_DIR}" --gt "${GT_DIR}" --output "${RESULTS_CSV}"
    [[ "$STEP" == "evaluate" ]] && exit 0
fi

echo "==== done: ${STEP} ===="
