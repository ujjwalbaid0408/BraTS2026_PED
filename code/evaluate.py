"""
Evaluation metrics for BraTS-PED 2026 segmentation.

Computes per-case and aggregate:
  - Dice Similarity Coefficient (DSC)
  - Normalized Surface Distance (NSD, tolerance = 1 mm)

BraTS-PED label scheme (dataset.json): background=0, ET=1, NET=2, CC=3, ED=4.
Evaluated regions (see EVAL_REGIONS in config.py; per BraTS-PED guidelines):
  ET  — Enhancing Tumor          (label 1)
  NET — Non-Enhancing Tumor core (label 2)
  CC  — Cystic Components        (label 3)
  ED  — Peritumoral Edema        (label 4)
  TC  — Tumor Core  = ET + NET + CC       (labels 1 + 2 + 3)
  WT  — Whole Tumor = ET + NET + CC + ED  (labels 1 + 2 + 3 + 4)

Usage:
    # Evaluate against ground truth:
    python evaluate.py --pred <pred_dir> --gt <gt_dir> --output results.csv

    # Evaluate post-processed predictions:
    python evaluate.py --pred <postproc_dir> --gt <gt_dir> --output results_pp.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, generate_binary_structure

sys.path.insert(0, str(Path(__file__).parent))
from config import EVAL_REGIONS, NSD_TOLERANCE_MM, CODES_DIR
from utils import get_logger

LOG = get_logger("evaluate", CODES_DIR / "logs" / "evaluate.log")


# ─── DSC ───────────────────────────────────────────────────────────────────────

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice Similarity Coefficient. Returns NaN if both masks are empty."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    if not p.any() and not g.any():
        return float("nan")   # Both empty: undefined (not penalised by challenge)
    if not p.any() or not g.any():
        return 0.0
    intersection = (p & g).sum()
    return 2.0 * intersection / (p.sum() + g.sum())


# ─── NSD ───────────────────────────────────────────────────────────────────────

def _surface_voxels(binary_mask: np.ndarray) -> np.ndarray:
    """Return a boolean array marking voxels on the surface of the mask."""
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(binary_mask, structure=struct, border_value=1)
    return binary_mask & ~eroded


def normalized_surface_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    tolerance_mm: float,
) -> float:
    """
    Normalized Surface Distance (NSD).

    Computes the fraction of gt surface points within `tolerance_mm` of any
    pred surface point, averaged with the symmetric fraction.
    Returns NaN if both masks are empty.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)

    if not p.any() and not g.any():
        return float("nan")
    if not p.any() or not g.any():
        return 0.0

    surf_p = _surface_voxels(p)
    surf_g = _surface_voxels(g)

    # Convert voxel indices to mm coordinates
    coords_p = np.argwhere(surf_p) * np.array(voxel_spacing)
    coords_g = np.argwhere(surf_g) * np.array(voxel_spacing)

    def _within_tolerance(src: np.ndarray, tgt: np.ndarray) -> float:
        """Fraction of src points within tolerance of the nearest tgt point."""
        # Batch distance computation: (N, 3) vs (M, 3)
        # For large surfaces this can be memory-intensive; chunked for safety.
        chunk = 2000
        within = 0
        for i in range(0, len(src), chunk):
            diffs = src[i:i+chunk, None, :] - tgt[None, :, :]   # (C, M, 3)
            dists = np.sqrt((diffs ** 2).sum(axis=2))            # (C, M)
            within += (dists.min(axis=1) <= tolerance_mm).sum()
        return within / len(src)

    nsd_p2g = _within_tolerance(coords_p, coords_g)
    nsd_g2p = _within_tolerance(coords_g, coords_p)
    return (nsd_p2g + nsd_g2p) / 2.0


# ─── Per-case evaluation ───────────────────────────────────────────────────────

def evaluate_case(
    pred_path: Path,
    gt_path: Path,
) -> Dict[str, float]:
    """
    Compute DSC and NSD for all EVAL_REGIONS for a single case.

    Returns a dict like:
        {"ETdsc": 0.78, "ETnsd": 0.61, "RCdsc": ..., ...}
    """
    pred_img = nib.load(str(pred_path))
    gt_img   = nib.load(str(gt_path))

    pred = np.asarray(pred_img.dataobj, dtype=np.int32)
    gt   = np.asarray(gt_img.dataobj,   dtype=np.int32)

    # Voxel spacing from affine diagonal
    voxel_spacing = tuple(np.abs(np.diag(pred_img.affine)[:3]).tolist())

    metrics = {}
    for region, label_vals in EVAL_REGIONS.items():
        pred_mask = np.isin(pred, label_vals)
        gt_mask   = np.isin(gt,   label_vals)

        dsc_val = dice(pred_mask, gt_mask)
        nsd_val = normalized_surface_distance(
            pred_mask, gt_mask, voxel_spacing, NSD_TOLERANCE_MM
        )
        metrics[f"{region}dsc"] = round(dsc_val, 4)
        metrics[f"{region}nsd"] = round(nsd_val, 4)

    return metrics


# ─── Folder evaluation ─────────────────────────────────────────────────────────

def evaluate_folder(
    pred_dir: Path,
    gt_dir: Path,
    output_csv: Optional[Path] = None,
) -> None:
    """
    Evaluate all predictions in pred_dir against ground truths in gt_dir.
    Prints per-case and aggregate results; optionally saves to CSV.
    """
    LOG.info("=== Evaluation ===")
    LOG.info("Predictions: %s", pred_dir)
    LOG.info("Ground truth: %s", gt_dir)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    pred_files = [f for f in pred_files if "probs" not in f.name]
    LOG.info("Found %d prediction files.", len(pred_files))

    region_keys = [f"{r}{m}" for r in EVAL_REGIONS for m in ("dsc", "nsd")]
    all_rows: List[Dict] = []

    for idx, pred_path in enumerate(pred_files, 1):
        # Match ground truth by case ID (strip nnU-Net channel suffix if present)
        case_id = pred_path.stem.replace(".nii", "")  # handle double extension
        gt_path = gt_dir / f"{case_id}.nii.gz"

        if not gt_path.exists():
            LOG.warning("GT not found for %s — skipping.", case_id)
            continue

        try:
            row = evaluate_case(pred_path, gt_path)
        except Exception as exc:
            LOG.error("Error evaluating %s: %s", case_id, exc)
            continue

        row["case_id"] = case_id
        all_rows.append(row)

        if idx % 50 == 0 or idx == len(pred_files):
            LOG.info("  Evaluated %d / %d", idx, len(pred_files))

    if not all_rows:
        LOG.error("No cases evaluated.")
        return

    # ── Aggregate stats (nanmean ignores NaN = both-empty cases) ──
    LOG.info("\n=== Aggregate Results (mean ± std) ===")
    summary = {}
    for key in region_keys:
        vals = [r[key] for r in all_rows if not np.isnan(r.get(key, float("nan")))]
        if vals:
            summary[key] = (np.mean(vals), np.std(vals))
            LOG.info("  %-8s  %.4f ± %.4f", key, *summary[key])

    # ── Save CSV ──
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["case_id"] + region_keys
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

            # Append summary rows
            writer.writerow({k: "" for k in fieldnames})
            mean_row = {"case_id": "MEAN"}
            std_row  = {"case_id": "STD"}
            for key in region_keys:
                if key in summary:
                    mean_row[key] = round(summary[key][0], 4)
                    std_row[key]  = round(summary[key][1], 4)
            writer.writerow(mean_row)
            writer.writerow(std_row)

        LOG.info("Results saved to: %s", output_csv)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BraTS-PED predictions.")
    p.add_argument("--pred",   required=True, type=Path, help="Directory with predicted .nii.gz files.")
    p.add_argument("--gt",     required=True, type=Path, help="Directory with ground-truth .nii.gz files.")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional CSV file to save per-case results.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_folder(args.pred, args.gt, args.output)
