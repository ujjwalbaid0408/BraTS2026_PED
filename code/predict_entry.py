"""
End-to-end inference entrypoint for the BraTS-PED Docker/Apptainer container.

Implements the BraTS-Lighthouse algorithm I/O contract:

    INPUT_DIR (default /input)
        BraTS-PED-XXXXX-000/
            BraTS-PED-XXXXX-000-t1n.nii.gz
            BraTS-PED-XXXXX-000-t1c.nii.gz
            BraTS-PED-XXXXX-000-t2w.nii.gz
            BraTS-PED-XXXXX-000-t2f.nii.gz
        ... (one folder per subject, NO segmentation, NO skull-stripping)

    OUTPUT_DIR (default /output)
        BraTS-PED-XXXXX-000.nii.gz       # predicted label map (1=ET,2=NET,3=CC,4=ED)

Pipeline per run:
    1. skull-strip every input case          (skull_strip.strip_case)
    2. rename to nnU-Net channels            (inference.convert_raw_to_nnunet)
    3. nnUNetv2_predict with the baked-in trained model (ensemble of folds)
    4. lesion-wise post-processing           (postprocess.postprocess_file)
    5. write OUTPUT_DIR/<case_id>.nii.gz

Model weights live in nnUNet_results (baked into the image at /opt/model and
exported via nnUNet_results). Folds and TTA are configurable via env vars.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODALITY_MAP, CASE_PREFIX, DATASET_ID, TRAINER, PLANNER, CONFIGURATION,
)
from utils import get_logger, discover_cases, build_nnunet_env, run_command
from skull_strip import strip_case, detect_backend
from inference import convert_raw_to_nnunet
from postprocess import postprocess_file

LOG = get_logger("predict_entry")

INPUT_DIR  = Path(os.environ.get("INPUT_DIR",  "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
FOLDS      = os.environ.get("BRATS_PED_INFER_FOLDS", "0").split()
USE_TTA    = os.environ.get("BRATS_PED_TTA", "0") == "1"
CHECKPOINT = os.environ.get("BRATS_PED_CHECKPOINT", "checkpoint_final.pth")
SS_METHOD  = os.environ.get("BRATS_PED_SS_METHOD", "auto")
SS_DEVICE  = os.environ.get("BRATS_PED_SS_DEVICE", "gpu")


def main() -> None:
    LOG.info("=== BraTS-PED container inference ===")
    LOG.info("INPUT_DIR=%s  OUTPUT_DIR=%s  folds=%s  tta=%s", INPUT_DIR, OUTPUT_DIR, FOLDS, USE_TTA)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = discover_cases([INPUT_DIR], MODALITY_MAP, CASE_PREFIX, require_seg=False)
    if not cases:
        LOG.error("No BraTS-PED cases found in %s", INPUT_DIR)
        sys.exit(1)
    LOG.info("Found %d case(s).", len(cases))

    work = Path(tempfile.mkdtemp(prefix="bratsped_work_"))
    ss_dir   = work / "stripped"
    nn_in    = work / "nnunet_in"
    nn_out   = work / "nnunet_out"
    ss_dir.mkdir(parents=True, exist_ok=True)

    backend = detect_backend(SS_METHOD)
    LOG.info("[1/4] Skull-stripping (backend=%s, device=%s)", backend, SS_DEVICE)
    for c in cases:
        strip_case(c, ss_dir, backend, SS_DEVICE)

    LOG.info("[2/4] Renaming to nnU-Net channel format")
    convert_raw_to_nnunet(ss_dir, nn_in)

    LOG.info("[3/4] nnU-Net prediction")
    env = build_nnunet_env(
        Path(os.environ["nnUNet_raw"]),
        Path(os.environ["nnUNet_preprocessed"]),
        Path(os.environ["nnUNet_results"]),
    )
    cmd = [
        "nnUNetv2_predict", "-i", str(nn_in), "-o", str(nn_out),
        "-d", str(DATASET_ID), "-c", CONFIGURATION, "-tr", TRAINER, "-p", PLANNER,
        "-f", *FOLDS, "-chk", CHECKPOINT,
    ]
    if not USE_TTA:
        cmd.append("--disable_tta")
    run_command(cmd, env, LOG)

    LOG.info("[4/4] Lesion-wise post-processing -> %s", OUTPUT_DIR)
    for pred in sorted(nn_out.glob("*.nii.gz")):
        if "probs" in pred.name:
            continue
        postprocess_file(pred, OUTPUT_DIR / pred.name)

    shutil.rmtree(work, ignore_errors=True)
    LOG.info("Done. Wrote predictions to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
