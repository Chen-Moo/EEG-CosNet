# -*- coding: utf-8 -*-
"""
Ablation study: verify the contribution of each innovation.

Via the ``use_cosine_conv`` and ``use_attention`` switches, four model
variants are constructed:

- ``baseline``        : standard convolution + attention (no cosine conv)
- ``cosine_only``     : cosine convolution + attention (standard conv removed)
- ``dropout_variant`` : cosine convolution + plain Dropout (replacing SpatialDropout)
- ``no_attention``    : cosine convolution + no attention (MHSA removed)

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


# Ablation variant configs: (name, use_cosine_conv, use_attention, dropout_type)
ABLATION_VARIANTS = [
    ('baseline',        False, True,  'SpatialDropout2D'),
    ('cosine_only',     True,  True,  'SpatialDropout2D'),
    ('dropout_variant', True,  True,  'Dropout'),
    ('no_attention',    True,  False, 'SpatialDropout2D'),
]


def run(data_dir=None, result_dir=None, variants=None, max_samples=5000):
    """Run the ablation study.

    Parameters
    ----------
    data_dir : str, optional
        Data directory; defaults to the ``EEG_DATA_DIR`` environment variable.
    result_dir : str, optional
        Result output directory; defaults to ``<RESULT_DIR>/ablation``.
    variants : list of str, optional
        Subset of variant names to run; defaults to all variants.
    max_samples : int
        Truncate samples to bound memory.
    """
    data_dir = data_dir or DATA_DIR

    mdd_path = os.path.join(data_dir, '19chans', 'MDDmix_total.npy')
    h_path = os.path.join(data_dir, '19chans', 'Hmix_total.npy')
    assert_file_exists(mdd_path)
    assert_file_exists(h_path)

    x, y = load_binary_dataset(mdd_path, h_path, max_samples=max_samples)
    chans = x.shape[1]
    samples = x.shape[2]

    result_root = result_dir or os.path.join(RESULT_DIR, 'ablation')

    selected = ABLATION_VARIANTS
    if variants:
        name_set = set(variants)
        selected = [v for v in ABLATION_VARIANTS if v[0] in name_set]

    for name, use_cos, use_attn, dropout_type in selected:
        output_dir = ensure_dir(os.path.join(result_root, name))

        def model_factory(use_cos=use_cos, use_attn=use_attn,
                          dt=dropout_type):
            return build_eeg_cosnet(
                chans=chans,
                samples=samples,
                dropout_rate=MODEL_DEFAULTS['dropout_rate'],
                dropout_type=dt,
                use_cosine_conv=use_cos,
                use_attention=use_attn,
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
            chans=chans,
            samples=samples,
        )
