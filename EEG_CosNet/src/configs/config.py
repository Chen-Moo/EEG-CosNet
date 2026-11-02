# -*- coding: utf-8 -*-
"""
Global configuration system.

All paths and hyperparameters are centralized here to avoid scattering
hard-coded absolute paths across scripts. Defaults target the open-source
scenario where data is not distributed with the repository: specify the local
data directory via the ``EEG_DATA_DIR`` environment variable or the
``DATA_DIR`` constant at the top of this file.

Dataset description
-------------------
This work uses two types of resting-state EEG data (see README):

- ``MDD*.npy`` -- Major Depressive Disorder patient samples
- ``H*.npy``   -- Healthy Control samples

Data can be obtained by contacting the authors or following the data source
declared in the paper.
"""

import os

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# Project root (this file lives at <repo>/src/configs/)
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directory: prefer the environment variable, fall back to ./data
DATA_DIR = os.environ.get('EEG_DATA_DIR',
                          os.path.join(PROJECT_ROOT, 'data'))

# Result output directory: model weights, metric tables, etc.
RESULT_DIR = os.environ.get('EEG_RESULT_DIR',
                            os.path.join(PROJECT_ROOT, 'results'))

# ---------------------------------------------------------------------------
# Dataset files (relative to DATA_DIR)
# ---------------------------------------------------------------------------
# Main experiment (7-channel frontal EEG, 1536 samples)
MAIN_MDD_FILE = os.path.join(DATA_DIR, 'MDDEC.npy')
MAIN_H_FILE = os.path.join(DATA_DIR, 'HEC.npy')

# Ablation / channel-decrease experiments (19-channel, mixed data)
ABLATION_MDD_FILE = os.path.join(DATA_DIR, '19chans', 'MDDmix_total.npy')
ABLATION_H_FILE = os.path.join(DATA_DIR, '19chans', 'Hmix_total.npy')

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL_MAP = {'MDD': 0, 'H': 1}

# ---------------------------------------------------------------------------
# Default model / training hyperparameters
# ---------------------------------------------------------------------------
MODEL_DEFAULTS = dict(
    nb_classes=2,
    chans=7,
    samples=1536,
    dropout_rate=0.2,
    kern_length=128,
    f1=8,
    depth_multiplier=2,
    f2=16,
    norm_rate=0.25,
    dropout_type='SpatialDropout2D',
    num_heads=2,
    key_dim=8,
)

TRAIN_DEFAULTS = dict(
    n_folds=6,
    epochs=150,
    batch_size=64,
    learning_rate=0.0005,
    random_state=42,
)


def ensure_dir(path):
    """Create the directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path
