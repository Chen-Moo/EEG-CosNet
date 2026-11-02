# -*- coding: utf-8 -*-
"""
Unified training and cross-validation engine.

Abstracts the highly duplicated K-Fold training / evaluation / metric-saving
flow from the original experiment scripts into a reusable
``run_cross_validation``. The main, ablation, and channel-decrease experiments
all train through this function, ensuring a consistent and reproducible
pipeline.
"""

import os
import gc
import numpy as np

from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import KFold

from src.utils.metrics import compute_metrics


def run_cross_validation(model_factory,
                         x,
                         y,
                         output_dir,
                         n_folds=6,
                         epochs=100,
                         batch_size=64,
                         learning_rate=0.0005,
                         random_state=42,
                         chans=None,
                         samples=None,
                         num_classes=2,
                         verbose=1):
    """Run K-Fold cross-validation and save per-fold metrics.

    Parameters
    ----------
    model_factory : callable
        A zero-argument callable returning a compiled model (used to rebuild
        the model for each fold).
    x : np.ndarray
        Feature array of shape ``(N, chans, samples)`` or
        ``(N, chans, samples, 1)``.
    y : np.ndarray
        Integer label array of shape ``(N,)``.
    output_dir : str
        Output directory (per-fold weights and metrics.txt are saved here).
    n_folds : int
        Number of folds.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    random_state : int
        Random seed (for reproducibility).
    chans : int, optional
        Number of channels for reshaping; ignored if ``x`` is already 4D.
    samples : int, optional
        Number of samples for reshaping.
    num_classes : int
        Number of classes.
    verbose : int
        Verbosity level passed to Keras training.

    Returns
    -------
    dict
        Summary containing per-fold metric lists and the average accuracy.
    """
    os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    acc, er, sn, sp, f1, auc = [], [], [], [], [], []
    fold_no = 1

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Uniformly reshape to (N, chans, samples, 1)
        if x_train.ndim == 3 and chans is not None and samples is not None:
            x_train = x_train.reshape(-1, chans, samples, 1)
            x_test = x_test.reshape(-1, chans, samples, 1)

        y_train_oh = to_categorical(y_train, num_classes=num_classes)
        y_test_oh = to_categorical(y_test, num_classes=num_classes)

        # Build and compile a fresh model for each fold
        model = model_factory()
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        save_name = os.path.join(output_dir, f'{fold_no}.h5')
        checkpointer = ModelCheckpoint(save_name,
                                       monitor='val_accuracy',
                                       verbose=verbose,
                                       save_best_only=True,
                                       mode='max')

        model.fit(x_train, y_train_oh,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test_oh),
                  callbacks=[checkpointer],
                  verbose=verbose)

        model.load_weights(save_name)
        _, accuracy = model.evaluate(x_test, y_test_oh, verbose=0)

        # Predictions for confusion matrix and AUC
        y_pred_prob = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test_oh, axis=1)

        metrics = compute_metrics(y_true, y_pred, y_score=y_pred_prob[:, 1])

        acc.append(accuracy)
        er.append(metrics['er'])
        sn.append(metrics['sn'])
        sp.append(metrics['sp'])
        f1.append(metrics['f1'])
        auc.append(metrics['auc'])

        print(f'Fold {fold_no}: ACC={accuracy:.4f} '
              f'Sn={metrics["sn"]:.4f} Sp={metrics["sp"]:.4f} '
              f'F1={metrics["f1"]:.4f} AUC={metrics["auc"]:.4f}')

        fold_no += 1

        del model
        gc.collect()
        K.clear_session()

    total_acc = float(np.mean(acc))

    # Save the per-fold metric table
    header = 'Fold\tACC\tEr\tSn\tSp\tF1\tAUC'
    rows = np.column_stack(
        ([str(i) for i in range(1, n_folds + 1)], acc, er, sn, sp, f1, auc))
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    np.savetxt(metrics_path, rows, fmt='%s', delimiter='\t', header=header)

    with open(metrics_path, 'a') as fp:
        fp.write(f'Average Accuracy: {total_acc:.6f}\n')

    print(f'Average Accuracy over {n_folds} folds: {total_acc:.6f}')
    print(f'Metrics saved to {metrics_path}')

    return {
        'acc': acc,
        'er': er,
        'sn': sn,
        'sp': sp,
        'f1': f1,
        'auc': auc,
        'total_acc': total_acc,
    }
