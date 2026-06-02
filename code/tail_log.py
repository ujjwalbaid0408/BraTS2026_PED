"""
Tail the most-recent nnU-Net training log in real time.

Finds the latest training_log_*.txt under nnUNet_results, then streams
new lines to stdout as they are written — similar to `tail -f` on Linux.

Usage:
    python tail_log.py                          # auto-detect latest log
    python tail_log.py --dataset-id 2           # specific dataset
    python tail_log.py --log <path/to/log.txt>  # explicit file
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import NNUNET_RESULTS, DATASET_ID


def find_latest_log(dataset_id: int) -> Path | None:
    dataset_dirs = sorted(NNUNET_RESULTS.glob(f"Dataset{dataset_id:03d}_*"))
    if not dataset_dirs:
        return None
    logs = sorted(dataset_dirs[-1].rglob("training_log_*.txt"),
                  key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def tail(log_path: Path) -> None:
    print(f"[tail_log] Streaming: {log_path}", flush=True)
    print("[tail_log] Press Ctrl+C to stop.\n", flush=True)

    with open(log_path, encoding="utf-8", errors="replace") as f:
        # Print everything already in the file first
        existing = f.read()
        if existing:
            print(existing, end="", flush=True)

        # Then follow new lines
        while True:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tail the latest nnU-Net training log.")
    p.add_argument("--dataset-id", type=int, default=DATASET_ID,
                   help=f"Dataset ID to search under (default: {DATASET_ID}).")
    p.add_argument("--log", type=Path, default=None,
                   help="Explicit path to training log file.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log_path = args.log
    if log_path is None:
        log_path = find_latest_log(args.dataset_id)
        if log_path is None:
            print(f"[tail_log] No training log found for Dataset{args.dataset_id:03d} "
                  f"under {NNUNET_RESULTS}")
            print("[tail_log] Start training first, then run this script.")
            sys.exit(1)

    try:
        tail(log_path)
    except KeyboardInterrupt:
        print("\n[tail_log] Stopped.")
