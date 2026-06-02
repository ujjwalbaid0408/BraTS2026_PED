# ============================================================================
#  BraTS-PED 2026 (Task 2) — inference container
#
#  Implements the BraTS-Lighthouse algorithm contract:
#     reads /input  (one folder per subject with t1n/t1c/t2w/t2f .nii.gz)
#     writes /output (<SubjectID>.nii.gz label maps; 1=ET 2=NET 3=CC 4=ED)
#
#  Pipeline inside the container: skull-strip -> nnU-Net predict -> postproc.
#
#  Build (from the PROJECT ROOT, so the build context contains ./code):
#     bash docker/build.sh                       # stages the model + builds
#  or manually:
#     docker build -f docker/Dockerfile -t bratsped2026:latest .
#
#  Run:
#     docker run --gpus all --rm \
#         -v /path/to/cases:/input:ro \
#         -v /path/to/out:/output \
#         bratsped2026:latest
#
#  CUDA 12.8 base => runs on RTX Pro 6000 (Blackwell) and older 12.x GPUs.
# ============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ── PyTorch (CUDA 12.8 wheels — Blackwell-compatible) ───────────────────────
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# ── Project code + dependencies (pip install -e .) ──────────────────────────
COPY code/ /opt/brats_ped/code/
RUN pip install -e /opt/brats_ped/code

# ── Trained model (staged by docker/build.sh into docker/model) ─────────────
#  Layout: /opt/model/Dataset021_BraTSPED2026/<trainer>__<planner>/3d_fullres/
COPY docker/model/ /opt/model/

# nnU-Net only needs nnUNet_results for inference; raw/preprocessed are dummies.
ENV nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_preprocessed \
    nnUNet_results=/opt/model \
    INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    BRATS_PED_INFER_FOLDS=0 \
    BRATS_PED_SS_METHOD=hdbet \
    BRATS_PED_SS_DEVICE=gpu \
    nnUNet_compile=f

RUN mkdir -p /tmp/nnunet_raw /tmp/nnunet_preprocessed /input /output

WORKDIR /opt/brats_ped/code
ENTRYPOINT ["python", "/opt/brats_ped/code/predict_entry.py"]
