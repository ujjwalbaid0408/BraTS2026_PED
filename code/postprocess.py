"""
Lesion-wise post-processing for BraTS-PED predictions.

The challenge is scored with the *lesion-wise* Dice / NSD, which evaluates each
connected lesion separately and penalises small false-positive components very
harshly. Pediatric sub-regions are highly imbalanced (ET present in ~67% of
training cases, CC ~35%, ED ~22%), so tiny spurious ET/CC/ED blobs are a common
and costly failure mode.

Following Chen & Liang (rank-2, BraTS-PED 2025), we relabel/remove tiny
connected components per class (POSTPROC_RULES in config.py):
    ET  components < 50  voxels  -> reassigned to NET
    CC  components < 150 voxels  -> reassigned to NET
    ED  components < 100 voxels  -> removed (background)

Usage:
    python postprocess.py --input <pred_dir> --output <postproc_dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from config import LABELS, POSTPROC_RULES, CODES_DIR
from utils import get_logger

LOG = get_logger("postprocess", CODES_DIR / "logs" / "postprocess.log")


def _apply_rule(data: np.ndarray, label_val: int, min_voxels: int, action: str) -> np.ndarray:
    """Relabel/remove connected components of `label_val` smaller than min_voxels."""
    if min_voxels <= 0:
        return data
    binary = data == label_val
    if not binary.any():
        return data

    labeled, n = ndimage.label(binary)
    if n == 0:
        return data
    sizes = ndimage.sum(np.ones_like(binary), labeled, index=range(1, n + 1))

    if action == "remove":
        new_val = 0
    elif action.startswith("reassign:"):
        new_val = int(action.split(":")[1])
    else:
        raise ValueError(f"Unknown action: {action}")

    out = data.copy()
    removed = 0
    for comp_idx, size in enumerate(sizes, start=1):
        if size < min_voxels:
            out[labeled == comp_idx] = new_val
            removed += 1
    if removed:
        LOG.debug("  label %d: %d/%d small components -> %s", label_val, removed, n, action)
    return out


def postprocess_file(pred_path: Path, out_path: Path) -> None:
    img = nib.load(str(pred_path))
    data = np.asarray(img.dataobj, dtype=np.int16)

    for name, rule in POSTPROC_RULES.items():
        data = _apply_rule(data, LABELS[name], rule["min_voxels"], rule["action"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header), str(out_path))


def postprocess_folder(input_dir: Path, output_dir: Path) -> None:
    LOG.info("=== Lesion-wise post-processing ===  %s -> %s", input_dir, output_dir)
    preds = [f for f in sorted(input_dir.glob("*.nii.gz")) if "probs" not in f.name]
    LOG.info("Found %d prediction files.", len(preds))
    for idx, p in enumerate(preds, 1):
        postprocess_file(p, output_dir / p.name)
        if idx % 25 == 0 or idx == len(preds):
            LOG.info("  %d / %d", idx, len(preds))
    LOG.info("Post-processing complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lesion-wise post-processing for BraTS-PED.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    postprocess_folder(args.input, args.output)


if __name__ == "__main__":
    main()
