# BraTS-PED 2026 (Task 2) — Pediatric Brain Tumour Segmentation

A reproducible, containerised pipeline for **Task 2 of the MICCAI BraTS-PED 2026
challenge**: voxel-wise segmentation of four pediatric brain-tumour sub-regions from
four co-registered MRI sequences.

The method distils the repeatedly-validated recipe of recent BraTS winners
(BraTS-METS 2025 GLI-team; BraTS-PED 2025 rank-1 Yi *et al.* and rank-2 Chen & Liang):

> **skull-strip → residual-encoder nnU-Net v2 (3D full-res, 5-fold ensemble) → lesion-wise post-processing**

| Label | Region | Present in (of 294 train) |
|------:|--------|---------------------------|
| 1 | Enhancing tumour (ET)      | 67 % |
| 2 | Non-enhancing tumour (NET) | 99 % |
| 3 | Cystic component (CC)      | 35 % |
| 4 | Peritumoral edema (ED)     | 22 % |

**Evaluation regions:** ET, NET, CC, ED, **TC** (1+2+3), **WT** (1+2+3+4).
**Metrics:** lesion-wise Dice (DSC) and Normalized Surface Distance (NSD).

---

## Repository layout

```
BraTS2026_PED/
├── code/                     # the pip-installable package (brats_ped)
│   ├── config.py             # all paths, labels, planner & post-proc rules
│   ├── skull_strip.py        # SynthStrip / HD-BET brain extraction
│   ├── prepare_dataset.py    # BraTS → nnU-Net v2 format (+ 10-case holdout)
│   ├── train.py              # plan / preprocess / train (resumable)
│   ├── inference.py          # nnU-Net prediction (ensemble + TTA)
│   ├── postprocess.py        # lesion-wise component relabel/removal
│   ├── evaluate.py           # DSC / NSD vs ground truth
│   ├── plot_training.py      # training/validation loss & Dice curves → PNG
│   ├── predict_entry.py      # full-chain container entrypoint
│   ├── setup.py / requirements.txt
│   ├── setup_env.sh / run_pipeline.sh
│   └── slurm/                # prep.sbatch, train.sbatch, submit_train.sh
├── docker/                   # Dockerfile + apptainer.def + build/run scripts
├── report/                   # LaTeX technical report (+ figure generator)
└── README.md
```

---

## 1. Install

```bash
# from the project root
bash code/setup_env.sh                 # CUDA 12.8 (RTX Pro 6000 / Blackwell)
# or, manually:
python3.11 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e code                    # installs ALL dependencies
```

`pip install -e code` installs nnU-Net v2, HD-BET, NiBabel, SimpleITK, SciPy,
scikit-image, matplotlib, pandas and the `brats_ped` package (console scripts
`bratsped-strip`, `bratsped-train`, `bratsped-infer`, …).

> **GPU note.** The pipeline targets a single **NVIDIA RTX Pro 6000 (96 GB)**.
> The H100/A100 partitions were intentionally avoided (decommissioned 2026-06-03).

---

## 2. Data

Point `code/config.py` (or the `BRATS_PED_TRAIN` / `BRATS_PED_VAL` env vars) at the
challenge data:

```
data/BraTS26_PED_training_Complete/BraTS-PED-XXXXX-000/
    BraTS-PED-XXXXX-000-{t1n,t1c,t2w,t2f}.nii.gz   +   -seg.nii.gz
data/BraTS26_PED_validation/BraTS-PED-XXXXX-000/    (no seg)
```

---

## 3. Train (SLURM, resumable)

```bash
# resource-aware submit: prep (skull-strip + preprocess) → train fold 0
bash code/slurm/submit_train.sh 0
# 5-fold ensemble (winning config):
bash code/slurm/submit_train.sh 0 1 2 3 4
```

The submission script inspects `sinfo`, picks an available **RTX Pro 6000** node
(`rp6b-1-gm96-c8-m64`, account `ai-gpu`, `--gpus=1`), and chains the jobs with
`afterok` dependencies.

**Checkpointing / resume.** nnU-Net saves `checkpoint_best.pth` (best EMA Dice) and
`checkpoint_latest.pth` (every 50 epochs). Jobs are `--requeue`-enabled and
auto-resume with `--c` after a timeout — *a GPU-timeout never loses more than ~50
epochs*. Training/validation curves are exported automatically by `plot_training.py`.

Run locally instead (no SLURM):

```bash
bash code/run_pipeline.sh all          # strip → prepare → preprocess → train → infer → postproc → evaluate
```

---

## 4. Inference on the validation set

```bash
source venv/bin/activate
python code/inference.py  --input data_ss/validation --output predictions/raw --raw-input --folds 0 1 2 3 4 --tta
python code/postprocess.py --input predictions/raw   --output predictions/postprocessed
```

---

## 5. Docker / Apptainer container (BraTS submission format)

The container reads `/input` (one folder per subject with the four modalities) and
writes `/output/<SubjectID>.nii.gz`. It skull-strips, predicts and post-processes
internally.

```bash
bash docker/build.sh                   # stages the trained model + builds image/SIF
bash docker/run_sample.sh              # runs on the 10 held-out cases + scores them
```

```bash
# manual run
docker run --gpus all --rm -v /path/cases:/input:ro -v /path/out:/output bratsped2026:latest
# or Apptainer (HPC)
apptainer run --nv --bind /path/cases:/input --bind /path/out:/output docker/bratsped2026.sif
```

---

## 6. Report

A LaTeX technical report is in `report/main.tex` (compile with `pdflatex`).
After training, `report/make_figures.py` regenerates the curves and qualitative
panels, and the results tables are populated from the held-out run.

---

## Method summary

1. **Skull-stripping** — BraTS-PED data retain skull/neck; we remove it (SynthStrip
   or HD-BET), applying the *identical* procedure at train and test time.
2. **nnU-Net v2 ResEnc-L, 3D full-res** — self-configuring residual-encoder U-Net,
   Dice + cross-entropy loss, SGD (Nesterov 0.99), lr 1e-2 poly decay, deep
   supervision, heavy augmentation.
3. **Five-fold ensemble** — softmax averaging; the largest accuracy lever on this
   small (294-case) dataset.
4. **Lesion-wise post-processing** — relabel tiny ET/CC components → NET, remove tiny
   ED components, directly targeting the lesion-wise metric.

## Acknowledgements & references

nnU-Net (Isensee *et al.*, *Nat. Methods* 2021); ResEnc presets (MIC-DKFZ 2024);
BraTS-PED challenge (Kazerooni *et al.*, arXiv:2404.15009); BraTS-PED 2025 rank-1
(Yi *et al.*) and rank-2 (Chen & Liang); SynthStrip (Hoopes *et al.*, *NeuroImage*
2022); HD-BET (Isensee *et al.* 2019).
