"""
Training orchestrator for BraTS-PED 2026 segmentation (nnU-Net v2).

Strategy:
  nnU-Net v2 | Residual-Encoder planner (ResEnc-L default) | 3d_fullres
  5-fold CV ensemble (or fold=all) | skull-stripped input | lesion-wise post-proc

Checkpointing / resume:
  nnU-Net automatically saves `checkpoint_best.pth` (best EMA pseudo-Dice) and
  `checkpoint_latest.pth` (every 50 epochs). If a run is killed by a GPU/time
  limit, restart with `--c` (or just rerun the SLURM script — it auto-detects
  the checkpoint) to continue from `checkpoint_latest.pth`.

Usage:
    python train.py --step all                     # preprocess + train fold 0
    python train.py --step preprocess              # plan & preprocess only
    python train.py --step train --folds 0 1 2 3 4 # 5-fold ensemble
    python train.py --step train --folds all --c   # resume fold=all
    python train.py --step train --folds 0 --trainer nnUNetTrainer_250epochs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS,
    DATASET_ID, TRAINER, PLANNER, PLANNER_CLASS, CONFIGURATION, FOLD, NUM_GPUS,
    CODES_DIR,
)
from utils import build_nnunet_env, run_command, get_logger

LOG = get_logger("train", CODES_DIR / "logs" / "train.log")


def step_preprocess(dataset_id: int, planner_class: str) -> None:
    LOG.info("=== Plan & Preprocess (dataset %d, planner %s) ===", dataset_id, planner_class)
    env = build_nnunet_env(NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS)
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-pl", planner_class,
        "--verify_dataset_integrity",
        "-c", CONFIGURATION,
    ]
    run_command(cmd, env, LOG)
    LOG.info("Preprocessing complete.")


def step_train(dataset_id: int, folds: list, trainer: str, planner: str,
               cont: bool, val_best: bool) -> None:
    LOG.info("=== Training (dataset %d, trainer %s, planner %s) ===", dataset_id, trainer, planner)
    env = build_nnunet_env(NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS)
    for fold in folds:
        LOG.info("--- Fold %s ---", fold)
        cmd = [
            "nnUNetv2_train", str(dataset_id), CONFIGURATION, str(fold),
            "-tr", trainer, "-p", planner,
        ]
        if NUM_GPUS > 1:
            cmd += ["-num_gpus", str(NUM_GPUS)]
        if cont:
            cmd.append("--c")          # continue from checkpoint_latest.pth
        if val_best:
            cmd.append("--val_best")   # also validate with checkpoint_best.pth
        run_command(cmd, env, LOG)
        LOG.info("Fold %s complete.", fold)
    LOG.info("All requested folds trained. Next: inference.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train nnU-Net for BraTS-PED segmentation.")
    p.add_argument("--step", choices=["all", "preprocess", "train"], default="all")
    p.add_argument("--dataset-id", type=int, default=DATASET_ID)
    p.add_argument("--trainer", default=TRAINER)
    p.add_argument("--planner", default=PLANNER)
    p.add_argument("--planner-class", default=PLANNER_CLASS)
    p.add_argument("--folds", nargs="+", default=[FOLD],
                   help="Folds to train: '0 1 2 3 4' (ensemble) or 'all'.")
    p.add_argument("--c", "--continue", dest="cont", action="store_true",
                   help="Continue from checkpoint_latest.pth (resume after timeout).")
    p.add_argument("--val-best", action="store_true",
                   help="Also run validation with checkpoint_best.pth.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.step in ("all", "preprocess"):
        step_preprocess(args.dataset_id, args.planner_class)
    if args.step in ("all", "train"):
        step_train(args.dataset_id, args.folds, args.trainer, args.planner,
                   args.cont, args.val_best)


if __name__ == "__main__":
    main()
