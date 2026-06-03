"""
Override the nnU-Net batch size in a generated plans file.

nnU-Net's ResEnc-L planner sizes the network for a ~24 GB VRAM budget, yielding
batch_size=2. On the RTX Pro 6000 (96 GB) we have ample headroom, so we raise the
batch size for smoother gradient estimates and better GPU utilisation. Changing the
batch size does NOT require re-running preprocessing (spacing/normalisation are
unchanged), so this can be applied any time before training starts.

Usage:
    python set_batch_size.py --batch-size 4
    python set_batch_size.py --batch-size 8 --config 3d_fullres
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import NNUNET_PREPROCESSED, DATASET_FULL_NAME, PLANNER, CONFIGURATION, CODES_DIR
from utils import get_logger

LOG = get_logger("set_batch_size", CODES_DIR / "logs" / "set_batch_size.log")


def main() -> None:
    p = argparse.ArgumentParser(description="Override nnU-Net batch size in plans.json.")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--config", default=CONFIGURATION)
    p.add_argument("--planner", default=PLANNER)
    args = p.parse_args()

    plans_path = NNUNET_PREPROCESSED / DATASET_FULL_NAME / f"{args.planner}.json"
    if not plans_path.exists():
        LOG.error("Plans file not found: %s (run planning first)", plans_path)
        sys.exit(1)

    plans = json.loads(plans_path.read_text())
    cfg = plans["configurations"][args.config]
    old = cfg.get("batch_size")
    cfg["batch_size"] = args.batch_size
    plans_path.write_text(json.dumps(plans, indent=2))
    LOG.info("patch_size = %s", cfg.get("patch_size"))
    LOG.info("batch_size: %s -> %s  in %s [%s]", old, args.batch_size, plans_path.name, args.config)


if __name__ == "__main__":
    main()
