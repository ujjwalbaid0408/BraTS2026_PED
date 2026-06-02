"""
Skull-stripping (brain extraction) for BraTS-PED data.

The BraTS-PED images are distributed WITHOUT skull-stripping. Both the rank-1
(Yi et al.) and rank-2 (Chen & Liang) BraTS-PED 2025 solutions report that the
skull/neck introduces non-biological variance that confounds training, and that
brain extraction is essential. We therefore strip every case the SAME way for
training, validation and inference (rank-1's consistent-stripping strategy).

Because the four modalities are already co-registered to a common space, we
compute ONE brain mask (from a robust modality) and apply it to all four
modalities. The segmentation label map (if present) is copied unchanged — all
tumour labels already lie inside the brain.

Backends (auto-selected, override with --method):
  synthstrip : FreeSurfer `mri_synthstrip` (contrast-agnostic, pediatric-robust;
               used by rank-2). Preferred when available on PATH.
  hdbet      : HD-BET (`pip install HD-BET`, MIC-DKFZ) — pip-installable,
               GPU/CPU, auto-downloads weights.
  none       : pass-through (copy only) — for pipeline smoke-tests.

Output layout mirrors the input (one folder per case):
    <out_root>/<case_id>/<case_id>-t1n.nii.gz ... -t2f.nii.gz [ -seg.nii.gz ]
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_TRAIN_ROOTS, RAW_VAL_ROOT, SS_TRAIN_ROOT, SS_VAL_ROOT,
    MODALITY_MAP, CASE_PREFIX, CODES_DIR,
)
from utils import discover_cases, get_logger

LOG = get_logger("skull_strip", CODES_DIR / "logs" / "skull_strip.log")

# Modality used to estimate the brain mask (native T1 has full FOV & good contrast).
MASK_MODALITY = "t1n"


def _resolve_exe(name: str) -> str:
    """Find a console script, preferring the one next to the active interpreter."""
    cand = Path(sys.executable).parent / name
    if cand.exists():
        return str(cand)
    found = shutil.which(name)
    return found if found else name


# ─── Backend detection ──────────────────────────────────────────────────────────

def detect_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if shutil.which("mri_synthstrip"):
        return "synthstrip"
    try:
        import HD_BET  # noqa: F401
        return "hdbet"
    except Exception:
        pass
    if shutil.which("hd-bet"):
        return "hdbet"
    LOG.warning("No skull-strip backend found (mri_synthstrip / HD-BET). "
                "Falling back to 'none' (pass-through). Install one for real stripping.")
    return "none"


# ─── Mask computation ───────────────────────────────────────────────────────────

def _mask_synthstrip(in_path: Path, mask_path: Path, device: str) -> None:
    import subprocess
    cmd = [_resolve_exe("mri_synthstrip"), "-i", str(in_path), "-m", str(mask_path)]
    if device == "cpu":
        cmd.append("--no-csf")  # harmless; keep CLI explicit
    subprocess.run(cmd, check=True)


def _mask_hdbet(in_path: Path, mask_path: Path, device: str) -> None:
    """Run HD-BET and produce a binary brain mask at mask_path."""
    import subprocess
    out_img = mask_path.parent / (mask_path.stem.replace(".nii", "") + "_hdbet.nii.gz")
    dev = "cuda" if device == "gpu" else "cpu"
    exe = _resolve_exe("hd-bet")
    # HD-BET v2 CLI: writes brain-extracted image + (with flag) the mask.
    cmd = [exe, "-i", str(in_path), "-o", str(out_img), "-device", dev, "--save_bet_mask"]
    if dev == "cpu":
        cmd.append("--disable_tta")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Older HD-BET CLI fallback.
        cmd = [exe, "-i", str(in_path), "-o", str(out_img), "-device", dev]
        subprocess.run(cmd, check=True)
    # HD-BET writes the mask next to the output with a _bet suffix.
    cand = list(out_img.parent.glob(out_img.stem.replace(".nii", "") + "*bet*.nii.gz"))
    if cand:
        shutil.move(str(cand[0]), str(mask_path))
    else:
        # Derive mask from the brain-extracted image (>0).
        img = nib.load(str(out_img))
        m = (np.asarray(img.dataobj) > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, img.affine, img.header), str(mask_path))
    if out_img.exists():
        out_img.unlink()


def compute_mask(in_path: Path, mask_path: Path, backend: str, device: str) -> None:
    if backend == "synthstrip":
        _mask_synthstrip(in_path, mask_path, device)
    elif backend == "hdbet":
        _mask_hdbet(in_path, mask_path, device)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ─── Per-case stripping ─────────────────────────────────────────────────────────

def strip_case(case: Dict, out_root: Path, backend: str, device: str) -> None:
    case_id = case["case_id"]
    out_dir = out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / ".stripped_ok"
    if done_flag.exists():
        return  # idempotent / resumable

    if backend == "none":
        for suffix, src in case["images"].items():
            shutil.copy2(src, out_dir / f"{case_id}-{suffix}.nii.gz")
        if case.get("seg"):
            shutil.copy2(case["seg"], out_dir / f"{case_id}-seg.nii.gz")
        done_flag.touch()
        return

    # 1. brain mask from the reference modality
    ref = case["images"][MASK_MODALITY]
    mask_path = out_dir / f"{case_id}-brainmask.nii.gz"
    compute_mask(ref, mask_path, backend, device)
    mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0

    # 2. apply mask to every modality
    for suffix, src in case["images"].items():
        img = nib.load(str(src))
        data = np.asarray(img.dataobj, dtype=np.float32)
        data[~mask] = 0
        nib.save(nib.Nifti1Image(data, img.affine, img.header),
                 str(out_dir / f"{case_id}-{suffix}.nii.gz"))

    # 3. copy segmentation unchanged (labels already inside brain)
    if case.get("seg"):
        shutil.copy2(case["seg"], out_dir / f"{case_id}-seg.nii.gz")

    done_flag.touch()


def strip_folder(root_dirs, out_root: Path, backend: str, device: str,
                 require_seg: bool, limit: Optional[int] = None) -> int:
    cases = discover_cases(root_dirs, MODALITY_MAP, CASE_PREFIX, require_seg=require_seg)
    if limit:
        cases = cases[:limit]
    LOG.info("Skull-stripping %d cases -> %s (backend=%s, device=%s)",
             len(cases), out_root, backend, device)
    out_root.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(cases, 1):
        try:
            strip_case(case, out_root, backend, device)
        except Exception as exc:
            LOG.error("FAILED %s: %s", case["case_id"], exc)
        if i % 10 == 0 or i == len(cases):
            LOG.info("  %d / %d done", i, len(cases))
    LOG.info("Done -> %s", out_root)
    return len(cases)


# ─── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Skull-strip BraTS-PED data.")
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--method", choices=["auto", "synthstrip", "hdbet", "none"], default="auto")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--limit", type=int, default=None, help="Strip only the first N cases (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    backend = detect_backend(args.method)
    LOG.info("=== Skull stripping (backend=%s) ===", backend)

    if args.split in ("train", "both"):
        strip_folder(RAW_TRAIN_ROOTS, SS_TRAIN_ROOT, backend, args.device,
                     require_seg=True, limit=args.limit)
    if args.split in ("val", "both"):
        strip_folder([RAW_VAL_ROOT], SS_VAL_ROOT, backend, args.device,
                     require_seg=False, limit=args.limit)


if __name__ == "__main__":
    main()
