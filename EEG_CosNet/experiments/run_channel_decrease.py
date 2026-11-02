# -*- coding: utf-8 -*-
"""
Channel-decrease study: examine recognition performance under different
brain-region / channel subsets.

The 19-channel EEG is partitioned by anatomical brain region and channel
count into several groups (G2~G7), each trained and evaluated with EEG-CosNet,
to analyze which brain regions contribute most to depression recognition and
whether the model retains performance under low-channel configurations.

Channel group definitions
-------------------------
- G2 (14ch): multi-region combination (central region removed)
- G3 (11ch): frontal + temporal
- G4 ( 9ch): frontal + occipital
- G5 ( 6ch): temporal + occipital
- G6 ( 7ch): frontal
- G7 ( 4ch): temporal

The data directory is expected to be provided via the ``EEG_DATA_DIR``
environment variable.
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


# Standard 19-channel arrangement (10-20 system)
TARGET_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz',
                   'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'Cz', 'Pz']

CHANNEL_TO_IDX = {ch: i for i, ch in enumerate(TARGET_CHANNELS)}

# Channel-decrease group configs
CHANNEL_GROUPS = {
    'G2': ['Fp1', 'F3', 'P3', 'O1', 'F7', 'T3', 'T5', 'Fz',
           'Fp2', 'F4', 'P4', 'O2', 'F8', 'T4', 'T6'],                 # 14ch
    'G3': ['Fp1', 'F3', 'F7', 'Fz', 'Fp2', 'F4', 'F8',
           'T3', 'T5', 'T4', 'T6'],                                    # 11ch
    'G4': ['Fp1', 'F3', 'F7', 'Fz', 'Fp2', 'F4', 'F8', 'O1', 'O2'],    # 9ch
    'G5': ['T3', 'T5', 'T4', 'T6', 'O1', 'O2'],                        # 6ch
    'G6': ['Fp1', 'F3', 'F7', 'Fz', 'Fp2', 'F4', 'F8'],                # 7ch
    'G7': ['T3', 'T5', 'T4', 'T6'],                                    # 4ch
}


def run(data_dir=None, result_dir=None, groups=None, max_samples=5000):
    """Run the channel-decrease study.

    Parameters
    ----------
    data_dir : str, optional
        Data directory; defaults to the ``EEG_DATA_DIR`` environment variable.
    result_dir : str, optional
        Result output directory; defaults to ``<RESULT_DIR>/channel_decrease``.
    groups : list of str, optional
        Subset of group names to run; defaults to all groups.
    max_samples : int
        Truncate samples to bound memory.
    """
    data_dir = data_dir or DATA_DIR

    mdd_path = os.path.join(data_dir, '19chans', 'MDDmix_total.npy')
    h_path = os.path.join(data_dir, '19chans', 'Hmix_total.npy')
    assert_file_exists(mdd_path)
    assert_file_exists(h_path)

    x_full, y = load_binary_dataset(mdd_path, h_path, max_samples=max_samples)

    result_root = result_dir or os.path.join(RESULT_DIR, 'channel_decrease')

    selected = CHANNEL_GROUPS
    if groups:
        selected = {k: v for k, v in CHANNEL_GROUPS.items()
                    if k in set(groups)}

    for gname, ch_names in selected.items():
        n_chans = len(ch_names)
        row_idx = [CHANNEL_TO_IDX[ch] for ch in ch_names]
        x = x_full[:, row_idx, :]  # (N, n_chans, T)

        output_dir = ensure_dir(os.path.join(result_root, gname))

        def model_factory(n_chans=n_chans, samples=x.shape[-1]):
            return build_eeg_cosnet(
                chans=n_chans,
                samples=samples,
                **{k: MODEL_DEFAULTS[k] for k in
                   ('dropout_rate', 'kern_length', 'f1', 'depth_multiplier',
                    'f2', 'norm_rate', 'dropout_type', 'num_heads', 'key_dim')},
            )

        run_cross_validation(
            model_factory=model_factory,
            x=x,
            y=y,
            output_dir=output_dir,
            n_folds=TRAIN_DEFAULTS['n_folds'],
            epochs=TRAIN_DEFAULTS['epochs'],
            batch_size=TRAIN_DEFAULTS['batch_size'],
            learning_rate=TRAIN_DEFAULTS['learning_rate'],
            chans=n_chans,
            samples=x.shape[-1],
        )
