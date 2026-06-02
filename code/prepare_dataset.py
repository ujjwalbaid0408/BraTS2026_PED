"""
Convert skull-stripped BraTS-PED training data to nnU-Net v2 format.

Reads the skull-stripped cases produced by skull_strip.py (SS_TRAIN_ROOT),
holds out N cases as an internal test set (never used for training — used for
the Docker demo and the report's quantitative/qualitative results), and writes
the remaining cases into nnU-Net v2 layout:

    nnUNet_raw/Dataset021_BraTSPED2026/
        imagesTr/BraTS-PED-XXXXX-000_0000.nii.gz  (t1n)
                 ..._0001 (t1c) _0002 (t2w) _0003 (t2f)
        labelsTr/BraTS-PED-XXXXX-000.nii.gz
        dataset.json

The held-out test case IDs are written to <PROJECT_ROOT>/holdout_test_cases.txt.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SS_TRAIN_ROOT, NNUNET_RAW, DATASET_FULL_NAME, MODALITY_MAP, LABELS,
    CASE_PREFIX, CODES_DIR, PROJECT_ROOT, N_HOLDOUT_TEST, HOLDOUT_SEED,
)
from utils import discover_cases, split_holdout, get_logger

LOG = get_logger("prepare_dataset", CODES_DIR / "logs" / "prepare_dataset.log")

HOLDOUT_FILE = PROJECT_ROOT / "holdout_test_cases.txt"


def build_dataset_json(out_dir: Path, num_training: int) -> None:
    channel_names = {str(i): list(MODALITY_MAP.keys())[i] for i in range(len(MODALITY_MAP))}
    dataset = {
        "channel_names": channel_names,
        "labels": {name: int(val) for name, val in LABELS.items()},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "BraTS-PED-2026",
        "description": "Pediatric brain tumour segmentation. "
                       "Labels: 1=ET, 2=NET, 3=CC, 4=ED.",
        "reference": "https://www.synapse.org/brats2025",
        "tensorImageSize": "3D",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset, indent=2))
    LOG.info("Wrote %s", out_dir / "dataset.json")


def prepare(use_symlinks: bool, n_holdout: int) -> None:
    LOG.info("=== BraTS-PED -> nnU-Net dataset preparation ===")
    cases = discover_cases([SS_TRAIN_ROOT], MODALITY_MAP, CASE_PREFIX, require_seg=True)
    LOG.info("Discovered %d skull-stripped training cases in %s", len(cases), SS_TRAIN_ROOT)
    if not cases:
        LOG.error("No cases found. Run skull_strip.py --split train first.")
        sys.exit(1)

    train_cases, holdout = split_holdout(cases, n_holdout, HOLDOUT_SEED)
    HOLDOUT_FILE.write_text("\n".join(sorted(c["case_id"] for c in holdout)) + "\n")
    LOG.info("Holdout internal-test cases (%d) written to %s", len(holdout), HOLDOUT_FILE)
    LOG.info("Training cases for nnU-Net: %d", len(train_cases))

    dataset_dir = NNUNET_RAW / DATASET_FULL_NAME
    images_dir, labels_dir = dataset_dir / "imagesTr", dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    transfer = (lambda s, d: d.symlink_to(s.resolve())) if use_symlinks else shutil.copy2

    for idx, case in enumerate(train_cases, 1):
        cid = case["case_id"]
        for suffix, ch in MODALITY_MAP.items():
            dest = images_dir / f"{cid}_{ch}.nii.gz"
            if not dest.exists():
                transfer(case["images"][suffix], dest)
        dest_seg = labels_dir / f"{cid}.nii.gz"
        if not dest_seg.exists():
            transfer(case["seg"], dest_seg)
        if idx % 50 == 0 or idx == len(train_cases):
            LOG.info("  %d / %d cases", idx, len(train_cases))

    build_dataset_json(dataset_dir, num_training=len(train_cases))
    LOG.info("Done. Next: train.py --step preprocess")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare BraTS-PED dataset for nnU-Net v2.")
    p.add_argument("--use-symlinks", action="store_true", help="Symlink instead of copy.")
    p.add_argument("--n-holdout", type=int, default=N_HOLDOUT_TEST,
                   help=f"Internal test cases to hold out (default {N_HOLDOUT_TEST}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepare(use_symlinks=args.use_symlinks, n_holdout=args.n_holdout)


if __name__ == "__main__":
    main()
