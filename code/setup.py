"""
Install the BraTS-PED segmentation project as an editable package:

    pip install -e .

Exposes all modules (config, utils, skull_strip, prepare_dataset, train,
inference, postprocess, evaluate, plot_training) plus console entry points.

PyTorch is intentionally NOT pinned here so you can install the wheel that
matches your CUDA (see setup_env.sh / README). nnunetv2 will pull a compatible
torch if one is not already present.
"""

from setuptools import setup, find_packages

setup(
    name="brats_ped",
    version="1.0.0",
    description="Pediatric Brain Tumour Segmentation — BraTS-PED 2026 (Task 2)",
    author="ujjwalbaid0408",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*", "docker*"]),
    py_modules=[
        "config", "utils", "skull_strip", "prepare_dataset",
        "train", "inference", "postprocess", "evaluate", "plot_training",
    ],
    install_requires=[
        "nnunetv2>=2.5.1",
        "nibabel>=5.2.0",
        "SimpleITK>=2.3.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "scikit-image>=0.22.0",
        "tqdm>=4.66.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        # Skull-stripping backend (pip-installable, MIC-DKFZ). SynthStrip via
        # FreeSurfer is preferred when present; HD-BET is the always-available
        # pip fallback so `pip install -e .` yields a fully working pipeline.
        "HD-BET>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "bratsped-strip       = skull_strip:main",
            "bratsped-prepare     = prepare_dataset:main",
            "bratsped-train       = train:main",
            "bratsped-infer       = inference:main",
            "bratsped-postproc    = postprocess:main",
            "bratsped-evaluate    = evaluate:main",
            "bratsped-plot        = plot_training:main",
        ],
    },
)
