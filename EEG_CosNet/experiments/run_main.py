# -*- coding: utf-8 -*-
"""
Main experiment: K-fold cross-validation of EEG-CosNet on frontal EEG for
depression recognition.

The data directory is expected to be provided via the ``EEG_DATA_DIR``
environment variable (the dataset is not distributed with this repository).
"""

import os

from src.configs.config import (
    DATA_DIR,
    RESULT_DIR,
    MODEL_DEFAULTS,
    TRAIN_DEFAULTS,
    ensure_dir,
)
from src.utils.data import load_binary_dataset, assert_file_exists
from src.utils.trainer import run_cross_validation
from src.models import build_eeg_cosnet


def run(data_dir=None, result_dir=None):
    """Run the main cross-validation experiment.

    Parameters
    ----------
    data_dir : str, optional
        Data directory; defaults to the ``EEG_DATA_DIR`` environment variable.
    result_dir : str, optional
        Result output directory; defaults to ``<RESULT_DIR>/main``.
    """
    data_dir = data_dir or DATA_DIR

    mdd_path = os.path.join(data_dir, 'MDDEC.npy')
    h_path = os.path.join(data_dir, 'HEC.npy')
    assert_file_exists(mdd_path)
    assert_file_exists(h_path)

    x, y = load_binary_dataset(mdd_path, h_path)
    chans = x.shape[1]
    samples = x.shape[2]

    result_dir = result_dir or os.path.join(RESULT_DIR, 'main')
    ensure_dir(result_dir)

    def model_factory():
        return build_eeg_cosnet(
            chans=chans,
            samples=samples,
            **{k: MODEL_DEFAULTS[k] for k in
               ('dropout_rate', 'kern_length', 'f1', 'depth_multiplier',
                'f2', 'norm_rate', 'dropout_type', 'num_heads', 'key_dim')},
        )

    run_cross_validation(
        model_factory=model_factory,
        x=x,
        y=y,
        output_dir=result_dir,
        n_folds=TRAIN_DEFAULTS['n_folds'],
        epochs=TRAIN_DEFAULTS['epochs'],
        batch_size=TRAIN_DEFAULTS['batch_size'],
        learning_rate=TRAIN_DEFAULTS['learning_rate'],
        chans=chans,
        samples=samples,
    )
