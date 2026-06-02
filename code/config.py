"""
Central configuration for the BraTS-PED 2026 (Task 2) segmentation project.

Pediatric brain tumour segmentation — adapted from the BraTS-METS winning
nnU-Net v2 pipeline (GLI-team, BraTS-METS 2025) and the BraTS-PED 2025
top-ranked solutions (Yi et al. rank-1; Chen & Liang rank-2), which all use
a self-configuring nnU-Net v2 backbone with skull-stripping + lesion-wise
post-processing.

Winning strategy implemented here:
  - Skull-strip every case (SynthStrip) BEFORE nnU-Net — pediatric data is
    NOT skull-stripped and the skull confounds training (rank-1 & rank-2).
  - nnU-Net v2, Residual-Encoder planner (ResEnc-L by default, XL optional),
    3d_fullres configuration.
  - 5-fold cross-validation ensemble (small dataset → ensembling helps a lot),
    or fold='all' for a single model.
  - Default CE + Dice loss, lr=0.01 with polynomial decay (nnU-Net default).
  - Lesion-wise post-processing: relabel/remove tiny ET / CC / ED components,
    which directly optimises the challenge's lesion-wise metric.

All paths are overridable via environment variables so the same code runs on
a workstation, a SLURM node, and inside the Docker/Apptainer container.
"""

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default


# ─── Project root ──────────────────────────────────────────────────────────────
# Repository directory (…/BraTS2026_PED/code). PROJECT_ROOT is its parent.
CODES_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = _env_path("BRATS_PED_ROOT", CODES_DIR.parent)

# ─── Raw challenge data ─────────────────────────────────────────────────────────
# Training has per-case ground-truth (*-seg.nii.gz); validation does NOT.
RAW_TRAIN_ROOTS = [
    _env_path("BRATS_PED_TRAIN",
              PROJECT_ROOT / "data" / "BraTS26_PED_training_Complete"),
]
RAW_VAL_ROOT = _env_path("BRATS_PED_VAL",
                         PROJECT_ROOT / "data" / "BraTS26_PED_validation")

# Backwards-compatible alias used by some helpers.
RAW_DATA_ROOTS = RAW_TRAIN_ROOTS

# ─── Skull-stripped (brain-extracted) copies ────────────────────────────────────
# Created by skull_strip.py; these feed nnU-Net for both training and inference.
SS_TRAIN_ROOT = _env_path("BRATS_PED_SS_TRAIN",
                          PROJECT_ROOT / "data_ss" / "training")
SS_VAL_ROOT   = _env_path("BRATS_PED_SS_VAL",
                          PROJECT_ROOT / "data_ss" / "validation")

# ─── nnU-Net workspace dirs ─────────────────────────────────────────────────────
NNUNET_BASE_DIR     = _env_path("BRATS_PED_NNUNET", PROJECT_ROOT / "nnUNet")
NNUNET_RAW          = _env_path("nnUNet_raw",          NNUNET_BASE_DIR / "nnUNet_raw")
NNUNET_PREPROCESSED = _env_path("nnUNet_preprocessed", NNUNET_BASE_DIR / "nnUNet_preprocessed")
NNUNET_RESULTS      = _env_path("nnUNet_results",      NNUNET_BASE_DIR / "nnUNet_results")

# ─── Dataset identity ──────────────────────────────────────────────────────────
DATASET_ID        = int(os.environ.get("BRATS_PED_DATASET_ID", "21"))
DATASET_NAME      = "BraTSPED2026"
DATASET_FULL_NAME = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

# Held-out internal test cases (never used for training) — used for the local
# Docker demo and the report's quantitative/qualitative results.
N_HOLDOUT_TEST = int(os.environ.get("BRATS_PED_N_HOLDOUT", "10"))
HOLDOUT_SEED   = 1337

# ─── nnU-Net training config ────────────────────────────────────────────────────
TRAINER       = os.environ.get("BRATS_PED_TRAINER", "nnUNetTrainer")  # 1000 epochs built-in
# Residual-encoder presets (nnU-Net >=2.4):
#   ResEncM  -> nnUNetResEncUNetMPlans   (fits ~9 GB,  fast)
#   ResEncL  -> nnUNetResEncUNetLPlans   (fits ~24 GB, strong — default here)
#   ResEncXL -> nnUNetResEncUNetXLPlans  (needs ~40 GB, strongest, slowest)
PLANNER_CLASS = os.environ.get("BRATS_PED_PLANNER_CLASS", "nnUNetPlannerResEncL")
PLANNER       = os.environ.get("BRATS_PED_PLANNER",       "nnUNetResEncUNetLPlans")
CONFIGURATION = os.environ.get("BRATS_PED_CONFIG", "3d_fullres")
# Default to single fold 0; use "all" for one model on all data, or 0..4 ensemble.
FOLD          = os.environ.get("BRATS_PED_FOLD", "0")
NUM_GPUS      = int(os.environ.get("BRATS_PED_NUM_GPUS", "1"))

# ─── Modality → nnU-Net channel index ──────────────────────────────────────────
# Order matches the standard BraTS convention used across all BraTS papers.
MODALITY_MAP = {
    "t1n": "0000",   # T1 native / pre-contrast
    "t1c": "0001",   # T1 post-contrast
    "t2w": "0002",   # T2 weighted
    "t2f": "0003",   # T2 FLAIR
}

# ─── Case-ID prefix ─────────────────────────────────────────────────────────────
CASE_PREFIX = "BraTS-PED-"

# ─── Segmentation labels (BraTS-PED) ────────────────────────────────────────────
LABELS = {
    "background": 0,
    "ET":  1,   # Enhancing tumour
    "NET": 2,   # Non-enhancing tumour
    "CC":  3,   # Cystic component
    "ED":  4,   # Peritumoral edema
}

# ─── Composite evaluation regions (challenge evaluation server) ─────────────────
# Six scored regions: the four base labels plus two composites.
#   TC (tumour core) = ET + NET + CC      (labels 1,2,3)
#   WT (whole tumour)= ET + NET + CC + ED (labels 1,2,3,4)
EVAL_REGIONS = {
    "ET":  [1],
    "NET": [2],
    "CC":  [3],
    "ED":  [4],
    "TC":  [1, 2, 3],
    "WT":  [1, 2, 3, 4],
}

# ─── Lesion-wise post-processing rules ──────────────────────────────────────────
# Pediatric tumours are highly imbalanced: ET present in only ~67% of cases,
# CC ~35%, ED ~22% (measured on the 294 training cases). The lesion-wise metric
# penalises small false-positive components harshly, so we relabel/remove tiny
# components (thresholds in voxels at 1 mm³ iso, following Chen & Liang rank-2).
#   action "remove"      -> set component to background (0)
#   action "reassign:N"  -> merge component into label N
POSTPROC_RULES = {
    "ET":  {"min_voxels": 50,  "action": "reassign:2"},  # tiny ET -> NET
    "CC":  {"min_voxels": 150, "action": "reassign:2"},  # tiny CC -> NET
    "ED":  {"min_voxels": 100, "action": "remove"},      # tiny ED -> background
    "NET": {"min_voxels": 0,   "action": "remove"},      # keep NET as-is
}

# ─── NSD surface-distance tolerance (mm) ───────────────────────────────────────
NSD_TOLERANCE_MM = 1.0
