"""
Run nnU-Net inference on a folder of BraTS-PED cases.

Input may be either:
  - nnU-Net channel format:  <id>_0000.nii.gz ... _0003.nii.gz   (default)
  - raw BraTS per-case dirs: <id>/<id>-t1n.nii.gz ...            (--raw-input)

For raw input the four modalities are renamed into nnU-Net channels in a temp
folder. NOTE: this script assumes the images are ALREADY skull-stripped (use
skull_strip.py first, or the Docker entrypoint which strips automatically).

Usage:
    python inference.py --input <ss_nnunet_dir> --output <out_dir>
    python inference.py --input <ss_val_dir> --output <out_dir> --raw-input --folds 0 1 2 3 4 --tta
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS,
    DATASET_ID, TRAINER, PLANNER, CONFIGURATION, FOLD, MODALITY_MAP,
    CASE_PREFIX, CODES_DIR,
)
from utils import build_nnunet_env, run_command, get_logger

LOG = get_logger("inference", CODES_DIR / "logs" / "inference.log")


def convert_raw_to_nnunet(raw_input_dir: Path, nnunet_input_dir: Path) -> None:
    """Rename raw BraTS per-case modality files into nnU-Net channel format."""
    nnunet_input_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = [d for d in sorted(raw_input_dir.iterdir())
                 if d.is_dir() and d.name.startswith(CASE_PREFIX)]
    LOG.info("Converting %d raw cases to nnU-Net input format...", len(case_dirs))
    for case_dir in case_dirs:
        cid = case_dir.name
        for suffix, ch in MODALITY_MAP.items():
            src = case_dir / f"{cid}-{suffix}.nii.gz"
            dest = nnunet_input_dir / f"{cid}_{ch}.nii.gz"
            if src.exists() and not dest.exists():
                shutil.copy2(src, dest)


def run_inference(input_dir: Path, output_dir: Path, folds: list, tta: bool,
                  save_probabilities: bool, dataset_id: int, trainer: str,
                  planner: str) -> None:
    LOG.info("=== Inference ===  input=%s output=%s folds=%s tta=%s",
             input_dir, output_dir, folds, tta)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = build_nnunet_env(NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir), "-o", str(output_dir),
        "-d", str(dataset_id), "-c", CONFIGURATION,
        "-tr", trainer, "-p", planner,
        "-f", *[str(f) for f in folds],
    ]
    if not tta:
        cmd.append("--disable_tta")
    if save_probabilities:
        cmd.append("--save_probabilities")
    run_command(cmd, env, LOG)
    LOG.info("Inference complete -> %s", output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nnU-Net inference for BraTS-PED.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--folds", nargs="+", default=[FOLD])
    p.add_argument("--dataset-id", type=int, default=DATASET_ID)
    p.add_argument("--trainer", default=TRAINER)
    p.add_argument("--planner", default=PLANNER)
    p.add_argument("--tta", action="store_true", help="Test-time augmentation (mirroring).")
    p.add_argument("--save-probabilities", action="store_true")
    p.add_argument("--raw-input", action="store_true",
                   help="Input is raw BraTS per-case dirs (already skull-stripped).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input
    if args.raw_input:
        tmp = Path(tempfile.mkdtemp(prefix="brats_ped_nnunet_in_"))
        convert_raw_to_nnunet(input_dir, tmp)
        input_dir = tmp
    run_inference(input_dir, args.output, args.folds, args.tta,
                  args.save_probabilities, args.dataset_id, args.trainer, args.planner)


if __name__ == "__main__":
    main()
