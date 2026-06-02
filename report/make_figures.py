"""
Generate qualitative + quantitative figures for the BraTS-PED report.

Outputs (into report/figures/):
  qualitative.png  — per-case T1C with predicted (and GT) label overlays
  metrics_bar.png  — per-region mean DSC bar chart from results.csv

Usage:
  python report/make_figures.py \
      --pred  docker/sample_run/output \
      --gt    docker/sample_run/gt \
      --images data/BraTS26_PED_training_Complete \
      --csv   docker/sample_run/results.csv
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# label -> RGBA (ET red, NET green, CC blue, ED yellow)
CMAP = ListedColormap([(0, 0, 0, 0), (0.85, 0.1, 0.1, 0.6),
                       (0.1, 0.7, 0.1, 0.6), (0.1, 0.3, 0.9, 0.6),
                       (0.95, 0.85, 0.1, 0.6)])


def _best_slice(seg: np.ndarray) -> int:
    """Axial slice with the most tumour voxels."""
    if (seg > 0).any():
        return int(np.argmax((seg > 0).sum(axis=(0, 1))))
    return seg.shape[2] // 2


def qualitative(pred_dir: Path, gt_dir: Path, images_dir: Path, out: Path, n: int = 4):
    preds = sorted(p for p in pred_dir.glob("*.nii.gz") if "probs" not in p.name)[:n]
    if not preds:
        print("No predictions found for qualitative figure.")
        return
    rows = len(preds)
    has_gt = gt_dir is not None and gt_dir.exists()
    cols = 3 if has_gt else 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_2d(axes)

    for r, pred_path in enumerate(preds):
        cid = pred_path.name.replace(".nii.gz", "")
        pred = np.asarray(nib.load(str(pred_path)).dataobj)
        t1c_path = images_dir / cid / f"{cid}-t1c.nii.gz"
        t1c = np.asarray(nib.load(str(t1c_path)).dataobj) if t1c_path.exists() else np.zeros_like(pred)
        gt = None
        if has_gt and (gt_dir / f"{cid}.nii.gz").exists():
            gt = np.asarray(nib.load(str(gt_dir / f"{cid}.nii.gz")).dataobj)
        z = _best_slice(gt if gt is not None else pred)

        def show(ax, overlay, title):
            ax.imshow(np.rot90(t1c[:, :, z]), cmap="gray")
            if overlay is not None:
                ax.imshow(np.rot90(overlay[:, :, z]), cmap=CMAP, vmin=0, vmax=4)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        show(axes[r, 0], None, f"{cid}\nT1C")
        show(axes[r, 1], pred, "Prediction")
        if has_gt:
            show(axes[r, 2], gt, "Ground truth")

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for c, l in
               [((0.85, 0.1, 0.1), "ET"), ((0.1, 0.7, 0.1), "NET"),
                ((0.1, 0.3, 0.9), "CC"), ((0.95, 0.85, 0.1), "ED")]]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)


def metrics_bar(csv_path: Path, out: Path):
    if not csv_path or not csv_path.exists():
        print("No results.csv — skipping metrics bar.")
        return
    import csv as _csv
    mean_row = None
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            if row.get("case_id") == "MEAN":
                mean_row = row
    if not mean_row:
        print("No MEAN row in results.csv — skipping.")
        return
    regions = ["ET", "NET", "CC", "ED", "TC", "WT"]
    vals = [float(mean_row.get(f"{r}dsc", "nan") or "nan") for r in regions]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(regions, vals, color="#225ea8")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Dice (held-out, n=10)")
    ax.set_title("Per-region Dice on internal test set")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--gt", type=Path, default=None)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "figures")
    args = p.parse_args()
    qualitative(args.pred, args.gt, args.images, args.out_dir / "qualitative.png")
    metrics_bar(args.csv, args.out_dir / "metrics_bar.png")


if __name__ == "__main__":
    main()
