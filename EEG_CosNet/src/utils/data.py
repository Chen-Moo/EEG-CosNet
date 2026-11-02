# -*- coding: utf-8 -*-
"""
Data processing and loading utilities.

Extracts the repeated "load data + append label + shuffle" flow from the
original scripts into reusable functions, shared by the main, ablation, and
channel-decrease experiments to reduce duplication and improve maintainability.
"""

import os
import numpy as np
from sklearn.utils import shuffle


def append_label(input_array, label):
    """Append a string label to each element of a sample array.

    Returns ``[(sample, label), ...]``.

    Parameters
    ----------
    input_array : array-like
        Sample array of shape ``(N, ...)``.
    label : str
        Class label to attach to each sample.

    Returns
    -------
    list of tuple
        ``[(sample, label), ...]``.
    """
    return [(item, label) for item in input_array]


def load_binary_dataset(mdd_path, h_path, label_map=None,
                        max_samples=None, allow_pickle=True,
                        shuffle_data=True, random_state=42):
    """Load a binary (MDD vs. healthy control) EEG dataset.

    Parameters
    ----------
    mdd_path : str
        Path to the ``.npy`` file of depression samples.
    h_path : str
        Path to the ``.npy`` file of healthy control samples.
    label_map : dict, optional
        Mapping from string label to integer, default ``{'MDD': 0, 'H': 1}``.
    max_samples : int, optional
        Truncate to the first ``max_samples`` samples (to bound memory).
    allow_pickle : bool
        Passed to ``np.load``.
    shuffle_data : bool
        Whether to shuffle the sample order.
    random_state : int
        Random seed.

    Returns
    -------
    x : np.ndarray
        Feature array.
    y : np.ndarray
        Integer label array.
    """
    if label_map is None:
        label_map = {'MDD': 0, 'H': 1}

    data_m = np.load(mdd_path, allow_pickle=allow_pickle)
    data_h = np.load(h_path, allow_pickle=allow_pickle)

    data_m = append_label(data_m, 'MDD')
    data_h = append_label(data_h, 'H')
    data_m.extend(data_h)

    data = shuffle(data_m, random_state=random_state) if shuffle_data else data_m
    if max_samples is not None:
        data = data[:max_samples]

    x = np.array([item[0] for item in data])
    y = np.array([label_map[item[1]] for item in data])

    return x, y


def assert_file_exists(path):
    """Assert a data file exists, otherwise raise an informative error.

    The open-source repository does not ship data files by default. When data
    is missing, this function gives a clear hint directing the user to set the
    ``EEG_DATA_DIR`` environment variable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Please set the EEG_DATA_DIR environment variable to the data "
            f"directory, or follow the data acquisition instructions in the "
            f"README.")
    return path
