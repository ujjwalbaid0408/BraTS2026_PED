"""
Parse nnU-Net training metrics and generate performance graphs.

Metrics are read from the checkpoint .pth file (logger state) — this is
the most reliable source. If no checkpoint exists yet, falls back to
parsing the plain-text training_log_*.txt file.

Output graphs (saved as PNG alongside the checkpoint):
  progress_custom.png  — train loss, val loss, pseudo Dice, EMA Dice
  lr_curve.png         — learning rate schedule
  epoch_times.png      — per-epoch wall-clock times

Usage:
    # Auto-detect from nnU-Net results folder (default: Dataset002, fold all)
    python plot_training.py

    # Explicit results folder (contains checkpoint_latest.pth)
    python plot_training.py --folder <path/to/fold_all>

    # Specify dataset/trainer for auto-detection
    python plot_training.py --dataset-id 2 --trainer nnUNetTrainer_100epochs
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    NNUNET_RESULTS, PLANNER, CONFIGURATION, FOLD, DATASET_ID, TRAINER, CODES_DIR,
)  # noqa: F401  (PLANNER/FOLD kept as module defaults)
from utils import get_logger

LOG = get_logger("plot_training", CODES_DIR / "logs" / "plot_training.log")


# ─── helpers ─────────────────────────────────────────────────────────────────

def _find_training_folder(dataset_id: int, trainer: str, planner: str, fold: str) -> Path:
    """Return the fold output folder inside nnUNet_results."""
    dataset_dirs = list(NNUNET_RESULTS.glob(f"Dataset{dataset_id:03d}_*"))
    if not dataset_dirs:
        LOG.error("No results folder found for dataset %d in %s", dataset_id, NNUNET_RESULTS)
        sys.exit(1)
    dataset_dir = dataset_dirs[0]

    # nnU-Net layout: <dataset>/<trainer>__<planner>__<config>/fold_<x>
    config_pattern = f"{trainer}__{planner}__{CONFIGURATION}/fold_{fold}"
    fold_dir = dataset_dir / config_pattern
    if not fold_dir.exists():
        LOG.error("Fold directory not found: %s", fold_dir)
        LOG.error("Has training started? Check nnUNet_results.")
        sys.exit(1)
    return fold_dir


def _load_from_checkpoint(folder: Path) -> dict | None:
    """Load logger dict from checkpoint_latest.pth (or checkpoint_best.pth)."""
    try:
        import torch
    except ImportError:
        LOG.warning("torch not importable; skipping checkpoint load.")
        return None

    for name in ("checkpoint_latest.pth", "checkpoint_best.pth", "checkpoint_final.pth"):
        ckpt_path = folder / name
        if ckpt_path.exists():
            LOG.info("Loading metrics from %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            logger_state = ckpt.get("logging", None)
            if logger_state and "train_losses" in logger_state:
                return logger_state
    return None


def _load_from_log_file(folder: Path) -> dict | None:
    """Parse training_log_*.txt as fallback metric source."""
    log_files = sorted(folder.glob("training_log_*.txt"))
    if not log_files:
        return None
    log_file = log_files[-1]  # most recent
    LOG.info("Parsing log file: %s", log_file)

    metrics = {
        "train_losses": [],
        "val_losses": [],
        "mean_fg_dice": [],
        "ema_fg_dice": [],
        "lrs": [],
        "epoch_durations": [],
    }

    text = log_file.read_text(encoding="utf-8", errors="replace")

    # Patterns emitted by nnUNetTrainer.on_epoch_end
    train_re   = re.compile(r"train_loss\s+([\-\d.eE+]+)")
    val_re     = re.compile(r"val_loss\s+([\-\d.eE+]+)")
    dice_re    = re.compile(r"Pseudo dice\s+\[([^\]]+)\]")
    ema_re     = re.compile(r"Yayy.*?EMA pseudo Dice:\s+([\d.]+)", re.IGNORECASE)
    time_re    = re.compile(r"Epoch time:\s+([\d.]+)\s+s")
    lr_re      = re.compile(r"Current lr:\s+([\d.eE+\-]+)")

    for m in train_re.finditer(text):
        metrics["train_losses"].append(float(m.group(1)))
    for m in val_re.finditer(text):
        metrics["val_losses"].append(float(m.group(1)))
    for m in dice_re.finditer(text):
        vals = [float(v.strip()) for v in m.group(1).split(",")]
        metrics["mean_fg_dice"].append(float(np.mean(vals)) if vals else float("nan"))
    for m in time_re.finditer(text):
        metrics["epoch_durations"].append(float(m.group(1)))
    for m in lr_re.finditer(text):
        metrics["lrs"].append(float(m.group(1)))

    if not metrics["train_losses"]:
        LOG.warning("No metrics parsed from log file.")
        return None
    return metrics


# ─── plotting ─────────────────────────────────────────────────────────────────

def _smooth(values: list, window: int = 5) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def plot_loss_and_dice(metrics: dict, out_path: Path, title_suffix: str = "") -> None:
    """4-panel figure: losses (raw + smooth), pseudo Dice, EMA Dice."""
    train_losses = metrics.get("train_losses", [])
    val_losses   = metrics.get("val_losses",   [])
    mean_dice    = metrics.get("mean_fg_dice",  [])
    ema_dice     = metrics.get("ema_fg_dice",   [])

    n_epochs = max(len(train_losses), len(val_losses), len(mean_dice))
    if n_epochs == 0:
        LOG.warning("No data to plot for loss/dice.")
        return
    epochs = list(range(n_epochs))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"nnU-Net Training Progress{title_suffix}", fontsize=14, fontweight="bold")

    # ── Panel 1: losses ──────────────────────────────────────────────────────
    ax = axes[0]
    if train_losses:
        ax.plot(epochs[:len(train_losses)], train_losses,
                alpha=0.35, color="steelblue", lw=1, label="train loss (raw)")
        ax.plot(epochs[:len(train_losses)], _smooth(train_losses),
                color="steelblue", lw=2, label="train loss (smooth)")
    if val_losses:
        ax.plot(epochs[:len(val_losses)], val_losses,
                alpha=0.35, color="tomato", lw=1, label="val loss (raw)")
        ax.plot(epochs[:len(val_losses)], _smooth(val_losses),
                color="tomato", lw=2, label="val loss (smooth)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (CE + Dice)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Dice ────────────────────────────────────────────────────────
    ax = axes[1]
    if mean_dice:
        ax.plot(epochs[:len(mean_dice)], mean_dice,
                alpha=0.4, color="seagreen", lw=1, label="mean fg Dice (raw)")
        ax.plot(epochs[:len(mean_dice)], _smooth(mean_dice),
                color="seagreen", lw=2, label="mean fg Dice (smooth)")
    if ema_dice:
        ax.plot(epochs[:len(ema_dice)], ema_dice,
                color="darkgreen", lw=2.5, ls="--", label="EMA Dice (moving avg)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pseudo Dice")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved: %s", out_path)


def plot_lr_curve(metrics: dict, out_path: Path, title_suffix: str = "") -> None:
    lrs = metrics.get("lrs", [])
    if not lrs:
        LOG.info("No LR data found — skipping lr_curve.png")
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f"Learning Rate Schedule{title_suffix}", fontsize=13, fontweight="bold")
    ax.plot(list(range(len(lrs))), lrs, color="darkorange", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved: %s", out_path)


def plot_epoch_times(metrics: dict, out_path: Path, title_suffix: str = "") -> None:
    # Checkpoint format stores timestamps; log-file fallback stores durations.
    if "epoch_start_timestamps" in metrics and "epoch_end_timestamps" in metrics:
        starts = metrics["epoch_start_timestamps"]
        ends   = metrics["epoch_end_timestamps"]
        n = min(len(starts), len(ends))
        durations = [ends[i] - starts[i] for i in range(n)]
    else:
        durations = metrics.get("epoch_durations", [])

    if not durations:
        LOG.info("No epoch timing data — skipping epoch_times.png")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f"Per-Epoch Training Time{title_suffix}", fontsize=13, fontweight="bold")
    ax.plot(list(range(len(durations))), durations, color="slateblue", lw=1.5, alpha=0.7)
    ax.axhline(np.mean(durations), color="navy", lw=2, ls="--",
               label=f"mean {np.mean(durations):.1f} s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Duration (s)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOG.info("Saved: %s", out_path)


# ─── main ─────────────────────────────────────────────────────────────────────

def make_plots(folder: Path, dataset_id: int, trainer: str) -> None:
    LOG.info("=== Plotting training metrics ===")
    LOG.info("Folder: %s", folder)

    metrics = _load_from_checkpoint(folder)
    if metrics is None:
        metrics = _load_from_log_file(folder)
    if metrics is None:
        LOG.error("No metrics found in %s. Start training first.", folder)
        sys.exit(1)

    n = len(metrics.get("train_losses", []))
    LOG.info("Epochs logged: %d", n)

    suffix = f" — Dataset{dataset_id:03d}, {trainer}, {folder.name}"
    plot_loss_and_dice(metrics, folder / "progress_custom.png", suffix)
    plot_lr_curve(metrics,     folder / "lr_curve.png",        suffix)
    plot_epoch_times(metrics,  folder / "epoch_times.png",     suffix)

    LOG.info("All plots written to: %s", folder)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot nnU-Net training metrics.")
    p.add_argument("--folder",     type=Path, default=None,
                   help="Path to fold_<x> directory containing checkpoints. "
                        "If omitted, auto-detected from --dataset-id and --trainer.")
    p.add_argument("--dataset-id", type=int,  default=DATASET_ID,
                   help=f"Dataset ID for auto-detection (default: {DATASET_ID}).")
    p.add_argument("--trainer",    default=TRAINER,
                   help=f"Trainer class name for auto-detection (default: {TRAINER}).")
    p.add_argument("--planner",    default=PLANNER,
                   help=f"Planner name for auto-detection (default: {PLANNER}).")
    p.add_argument("--fold",       default=FOLD,
                   help=f"Fold for auto-detection (default: {FOLD}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.folder
    if folder is None:
        folder = _find_training_folder(args.dataset_id, args.trainer, args.planner, args.fold)
    make_plots(folder, args.dataset_id, args.trainer)


if __name__ == "__main__":
    main()
