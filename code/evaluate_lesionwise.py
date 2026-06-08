"""
Lesion-wise evaluation for BraTS-PED 2026 — faithful to the official BraTS-2023
metric (github.com/rachitsaluja/BraTS-2023-Metrics, the implementation used by
the challenge's Synapse evaluation server).

For each scored region the metric:
  1. Binarises GT and prediction for the region.
  2. Connected-component analysis (cc3d, 26-connectivity).
  3. Merges GT components whose dilations overlap into single "lesions"
     (dilation_factor=3, structuring element generate_binary_structure(3,2)).
  4. Per GT lesion: dilate it, take the UNION of predicted components that
     intersect the dilation as the matched prediction, and score lesion-wise
     Dice + HD95 against the (un-dilated) GT lesion.
  5. Predicted components matching no GT lesion are false positives (FP).
  6. GT lesions with volume <= lesion_volume_thresh (50 mm^3) are dropped.
  7. Aggregate:
        LesionWise_Dice = sum(dice_l) / (n_kept_lesions + n_fp)
        LesionWise_HD95 = (sum(hd95_l) + n_fp*374) / (n_kept_lesions + n_fp)
     A missing GT lesion contributes dice 0 and HD95 374 (the FP/FN penalty).

BraTS-PED label scheme: ET=1, NET=2, CC=3, ED=4. Scored regions come from
EVAL_REGIONS in config.py (ET, NET, CC, ED, TC=1+2+3, WT=1+2+3+4).

Usage:
    python evaluate_lesionwise.py --pred <pred_dir> --gt <gt_dir> --output lw.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import cc3d
import scipy.ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from config import EVAL_REGIONS, CODES_DIR
from utils import get_logger

LOG = get_logger("evaluate_lw", CODES_DIR / "logs" / "evaluate_lesionwise.log")

# ── Official BraTS-PED parameters ───────────────────────────────────────────
DILATION_FACTOR = 3        # iterations of binary dilation (BraTS-PED/GLI/SSA)
LESION_VOLUME_THRESH = 50  # mm^3 — GT lesions at/below this are ignored
HD95_PENALTY = 374.0       # HD95 assigned to a FP/FN lesion (inf -> 374)

# ── HD95 backend: prefer DeepMind surface_distance; fall back to scipy EDT ───
try:
    import surface_distance as _sd  # noqa

    def _hd95(gt_bin: np.ndarray, pred_bin: np.ndarray, spacing) -> float:
        g, p = gt_bin.astype(bool), pred_bin.astype(bool)
        if not g.any() and not p.any():
            return 0.0
        if not g.any() or not p.any():
            return math.inf
        sd = _sd.compute_surface_distances(g, p, spacing)
        return _sd.compute_robust_hausdorff(sd, 95)

    HD_BACKEND = "surface_distance"
except Exception:  # pragma: no cover - dependency optional
    def _surface(mask: np.ndarray) -> np.ndarray:
        st = ndi.generate_binary_structure(3, 1)
        return mask & ~ndi.binary_erosion(mask, structure=st, border_value=0)

    def _hd95(gt_bin: np.ndarray, pred_bin: np.ndarray, spacing) -> float:
        g, p = gt_bin.astype(bool), pred_bin.astype(bool)
        if not g.any() and not p.any():
            return 0.0
        if not g.any() or not p.any():
            return math.inf
        gs, ps = _surface(g), _surface(p)
        d_to_p = ndi.distance_transform_edt(~ps, sampling=spacing)
        d_to_g = ndi.distance_transform_edt(~gs, sampling=spacing)
        d_g2p = d_to_p[gs]
        d_p2g = d_to_g[ps]
        return float(max(np.percentile(d_g2p, 95), np.percentile(d_p2g, 95)))

    HD_BACKEND = "scipy-edt"


def _dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p, g = pred_bin.astype(bool), gt_bin.astype(bool)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    return 2.0 * (p & g).sum() / (p.sum() + g.sum())


def _crop(*masks, margin: int) -> Tuple[slice, ...]:
    """Bounding box (with margin) around the union of the given masks."""
    union = np.zeros(masks[0].shape, dtype=bool)
    for m in masks:
        union |= m
    if not union.any():
        return tuple(slice(0, 1) for _ in range(union.ndim))
    coords = np.argwhere(union)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, union.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def lesionwise_region(gt_bin: np.ndarray, pred_bin: np.ndarray, spacing) -> Dict[str, float]:
    """Lesion-wise Dice/HD95 for one binary region (official BraTS-2023 logic)."""
    voxvol = float(np.prod(spacing))
    struct = ndi.generate_binary_structure(3, 2)

    # Empty-GT shortcuts (matches official: no lesions -> perfect if pred empty)
    if not gt_bin.any() and not pred_bin.any():
        return {"dice": 1.0, "hd95": 0.0, "n_tp": 0, "n_fp": 0, "n_fn": 0}

    # Crop to the region's bounding box (+margin) for speed; safe because all
    # lesions live inside it and dilation stays within the margin.
    sl = _crop(gt_bin, pred_bin, margin=DILATION_FACTOR + 2)
    gt_bin = gt_bin[sl]
    pred_bin = pred_bin[sl]

    gt_cc = cc3d.connected_components(gt_bin.astype(np.uint8), connectivity=26)
    pred_cc = cc3d.connected_components(pred_bin.astype(np.uint8), connectivity=26)

    # Combine GT components whose dilations overlap into single lesions.
    gt_dil = ndi.binary_dilation(gt_bin, structure=struct, iterations=DILATION_FACTOR)
    gt_dil_cc = cc3d.connected_components(gt_dil.astype(np.uint8), connectivity=26)
    gt_comb = np.zeros_like(gt_dil_cc)
    for comp in range(1, int(gt_dil_cc.max()) + 1):
        tmp = gt_cc * (gt_dil_cc == comp)
        gt_comb[tmp > 0] = comp

    tp_labels: List[int] = []
    fn_count = 0
    pairs: List[Tuple[int, float, float, float]] = []  # (n_inter, gt_vol, dice, hd95)

    for gtcomp in range(1, int(gt_comb.max()) + 1):
        gt_tmp = gt_comb == gtcomp
        if not gt_tmp.any():
            continue
        gt_vol = float(gt_tmp.sum()) * voxvol
        gt_tmp_dil = ndi.binary_dilation(gt_tmp, structure=struct, iterations=DILATION_FACTOR)
        inter = np.unique(pred_cc * gt_tmp_dil)
        inter = inter[inter != 0]
        tp_labels.extend(int(c) for c in inter)

        pred_tmp = np.isin(pred_cc, inter)  # union of matched predicted components
        dsc = _dice(pred_tmp, gt_tmp)
        hd = _hd95(gt_tmp, pred_tmp, spacing)
        pairs.append((len(inter), gt_vol, dsc, hd))
        if len(inter) == 0:
            fn_count += 1

    fp_labels = np.unique(pred_cc[~np.isin(pred_cc, tp_labels + [0])])
    n_fp = int(len(fp_labels))

    # Drop GT lesions at/below the volume threshold; keep the rest (incl. FNs).
    kept = [(d, h) for (_n, vol, d, h) in pairs if vol > LESION_VOLUME_THRESH]
    n_fn_kept = sum(1 for (_n, vol, _d, _h) in pairs if vol > LESION_VOLUME_THRESH and _n == 0)
    n_tp_kept = sum(1 for (_n, vol, _d, _h) in pairs if vol > LESION_VOLUME_THRESH and _n != 0)

    denom = len(kept) + n_fp
    if denom == 0:
        # No GT lesions above threshold and no FP: nothing to score.
        return {"dice": 1.0, "hd95": 0.0, "n_tp": 0, "n_fp": 0, "n_fn": 0}

    sum_dice = sum(d for d, _h in kept)
    sum_hd = sum((HD95_PENALTY if math.isinf(h) else h) for _d, h in kept) + n_fp * HD95_PENALTY
    return {
        "dice": sum_dice / denom,
        "hd95": sum_hd / denom,
        "n_tp": n_tp_kept,
        "n_fp": n_fp,
        "n_fn": n_fn_kept,
    }


def evaluate_case(pred_path: Path, gt_path: Path) -> Dict[str, float]:
    pred_img = nib.load(str(pred_path))
    gt_img = nib.load(str(gt_path))
    pred = np.asarray(pred_img.dataobj, dtype=np.int16)
    gt = np.asarray(gt_img.dataobj, dtype=np.int16)
    spacing = tuple(float(z) for z in pred_img.header.get_zooms()[:3])

    row: Dict[str, float] = {}
    for region, labels in EVAL_REGIONS.items():
        # NB: lesion-wise is asymmetric — GT must be the first argument.
        res = lesionwise_region(np.isin(gt, labels), np.isin(pred, labels), spacing)
        row[f"{region}_LWdice"] = round(res["dice"], 4)
        row[f"{region}_LWhd95"] = round(res["hd95"], 4)
    return row


def evaluate_folder(pred_dir: Path, gt_dir: Path, output_csv: Path = None) -> None:
    LOG.info("=== Lesion-wise evaluation (backend=%s) ===", HD_BACKEND)
    LOG.info("pred=%s gt=%s", pred_dir, gt_dir)
    pred_files = [f for f in sorted(pred_dir.glob("*.nii.gz")) if "probs" not in f.name]
    region_keys = [f"{r}_{m}" for r in EVAL_REGIONS for m in ("LWdice", "LWhd95")]
    rows: List[Dict] = []

    for idx, pf in enumerate(pred_files, 1):
        cid = pf.name[:-7]
        gp = gt_dir / f"{cid}.nii.gz"
        if not gp.exists():
            LOG.warning("GT missing for %s — skipping", cid)
            continue
        try:
            r = evaluate_case(pf, gp)
        except Exception as exc:  # pragma: no cover
            LOG.error("Error on %s: %s", cid, exc)
            continue
        r["case_id"] = cid
        rows.append(r)
        if idx % 25 == 0 or idx == len(pred_files):
            LOG.info("  %d / %d", idx, len(pred_files))

    if not rows:
        LOG.error("No cases evaluated.")
        return

    LOG.info("\n=== Aggregate lesion-wise (mean) ===")
    summary = {}
    for k in region_keys:
        vals = [r[k] for r in rows if not (isinstance(r.get(k), float) and math.isnan(r.get(k)))]
        if vals:
            summary[k] = (float(np.mean(vals)), float(np.std(vals)))
            LOG.info("  %-14s %.4f ± %.4f", k, *summary[k])

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["case_id"] + region_keys
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
            w.writerow({k: "" for k in fieldnames})
            mean_row = {"case_id": "MEAN"}
            for k in region_keys:
                if k in summary:
                    mean_row[k] = round(summary[k][0], 4)
            w.writerow(mean_row)
        LOG.info("Saved -> %s", output_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lesion-wise BraTS-PED evaluation (Dice + HD95).")
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate_folder(a.pred, a.gt, a.output)
