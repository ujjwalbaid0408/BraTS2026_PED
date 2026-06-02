"""
Shared utilities: logging, case discovery, holdout split, nnU-Net env, subprocess.
"""

import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ─── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a logger that writes to stdout (and optionally a file)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─── Case discovery ────────────────────────────────────────────────────────────

def discover_cases(
    root_dirs: List[Path],
    modality_map: Dict[str, str],
    case_prefix: str = "BraTS-PED-",
    require_seg: bool = True,
) -> List[Dict]:
    """
    Scan root_dirs for BraTS case folders.

    A valid case folder contains one file per modality suffix (t1n, t1c, t2w,
    t2f) and, if require_seg, a segmentation file ending in -seg.nii.gz.

    Returns a list of dicts, one per case:
        {"case_id": "BraTS-PED-00001-000",
         "images":  {"t1n": Path, "t1c": Path, "t2w": Path, "t2f": Path},
         "seg":     Path | None}
    """
    cases: List[Dict] = []
    seen_ids = set()

    for root in root_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir() or not candidate.name.startswith(case_prefix):
                continue
            case_id = candidate.name
            if case_id in seen_ids:
                continue

            images: Dict[str, Path] = {}
            for suffix in modality_map:
                matches = list(candidate.glob(f"*-{suffix}.nii.gz"))
                if len(matches) == 1:
                    images[suffix] = matches[0]

            seg_matches = list(candidate.glob("*-seg.nii.gz"))
            seg = seg_matches[0] if len(seg_matches) == 1 else None

            if len(images) == len(modality_map) and (not require_seg or seg is not None):
                cases.append({"case_id": case_id, "images": images, "seg": seg})
                seen_ids.add(case_id)

    return cases


def split_holdout(cases: List[Dict], n_holdout: int, seed: int) -> tuple:
    """Deterministically split cases into (train_cases, holdout_test_cases)."""
    ordered = sorted(cases, key=lambda c: c["case_id"])
    rng = random.Random(seed)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    holdout_idx = set(idx[:n_holdout])
    train = [c for i, c in enumerate(ordered) if i not in holdout_idx]
    holdout = [c for i, c in enumerate(ordered) if i in holdout_idx]
    return train, holdout


# ─── nnU-Net environment ────────────────────────────────────────────────────────

def build_nnunet_env(nnunet_raw: Path, nnunet_preprocessed: Path, nnunet_results: Path) -> Dict[str, str]:
    """Return an os.environ-compatible dict with nnU-Net path variables set."""
    env = os.environ.copy()
    env["nnUNet_raw"]          = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    env["nnUNet_results"]      = str(nnunet_results)

    venv_scripts = Path(sys.executable).parent
    if venv_scripts.exists():
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")

    env["PYTHONUNBUFFERED"] = "1"
    return env


# ─── Subprocess runner ─────────────────────────────────────────────────────────

def run_command(cmd: List[str], env: Dict[str, str], logger: logging.Logger, check: bool = True) -> int:
    """Run a command, stream stdout/stderr to logger, and return exit code."""
    import shutil

    exe = cmd[0]
    resolved = shutil.which(exe, path=env.get("PATH", os.environ.get("PATH", "")))
    if resolved:
        cmd = [resolved] + list(cmd[1:])

    logger.info("Running: %s", " ".join(map(str, cmd)))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    for line in proc.stdout:
        logger.info(line.rstrip())
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(map(str, cmd))}")
    return proc.returncode
