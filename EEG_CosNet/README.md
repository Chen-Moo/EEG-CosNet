# EEG-CosNet

A **cosine-convolution + multi-head self-attention** network for EEG-based
depression recognition.

This repository is the official code implementation of the paper *"[Paper
Title]"*, which builds on the classical EEGNet with two improvements and
validates their effectiveness on frontal resting-state EEG for depression
recognition.

---

## Contributions

### 1. Cosine Convolution Kernel

A standard convolution ``sum(x*w) + b`` scales its response linearly with
input amplitude, while EEG signals fluctuate strongly in absolute amplitude
across subjects and trials. We propose replacing the inner product with
**L2 normalization + cosine similarity**:

```
y = (x / ||x||) . (w / ||w||)
```

so that the convolution response is determined only by the feature
*direction* (i.e. waveform shape) and is insensitive to amplitude scaling,
better matching the discriminative characteristics of EEG signals. This layer
(`CosineConv2D`) is fully API-compatible with `keras.layers.Conv2D` and can
replace it painlessly.

### 2. Spatial Dropout + Multi-Head Self-Attention

After each depthwise convolution block, we insert a composite module of
**Spatial Dropout** and **multi-head self-attention**. Spatial Dropout
deactivates entire feature maps as a unit, matching the spatial correlation of
convolutional feature maps; multi-head self-attention models long-range
dependencies along the time dimension, compensating for the limited receptive
field of local convolutions. Together they notably **stabilize training**,
accelerate convergence, and yield consistent accuracy gains.

---

## Directory Structure

```
EEG-CosNet/
├── src/
│   ├── layers/              # custom layers
│   │   ├── cosine_conv.py   #   cosine convolution layer CosineConv2D
│   │   └── attention.py     #   spatial dropout + multi-head attention module
│   ├── models/
│   │   └── eeg_cosnet.py    # EEG-CosNet model (with ablation switches)
│   ├── configs/
│   │   └── config.py        # global config (paths / hyperparameters)
│   └── utils/
│       ├── data.py          # data loading
│       ├── metrics.py       # evaluation metrics
│       └── trainer.py       # K-Fold cross-validation engine
├── experiments/
│   ├── run_main.py          # main experiment
│   ├── run_ablation.py      # ablation study
│   └── run_channel_decrease.py  # channel-decrease study
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
conda create -n eeg-cosnet python=3.9
conda activate eeg-cosnet
pip install -r requirements.txt
```

> Dependencies: TensorFlow >= 2.x (with Keras), numpy, scikit-learn.

---

## Data Preparation

This repository does **not** ship the dataset. Please obtain the data
according to the source declared in the paper and place it under the `data/`
directory (or specify it via the `EEG_DATA_DIR` environment variable):

```
data/
├── MDDEC.npy            # main experiment: depression samples (frontal, 7ch)
├── HEC.npy              # main experiment: healthy control samples
└── 19chans/
    ├── MDDmix_total.npy # ablation / channel-decrease: depression (19ch)
    └── Hmix_total.npy   # ablation / channel-decrease: healthy control
```

Data format: numpy arrays of shape `(N, chans, samples)`.

---

## Reproducing the Experiments

The training pipeline is organized into three experiment modules under
`experiments/`, each exposing a `run()` function driven by the centralized
configuration in `src/configs/config.py` and the shared cross-validation
engine in `src/utils/trainer.py`:

- `run_main.py` — the main experiment (frontal 7-channel EEG, 6-fold CV).
- `run_ablation.py` — the ablation study over the four model variants.
- `run_channel_decrease.py` — the channel-decrease study across brain-region
  subsets (G2~G7).

The ablation variants map onto the two contributions as follows:

| Variant | Cosine conv | MHSA | Description |
|---------|:-----------:|:----:|-------------|
| `baseline` | no | yes | standard conv + attention (control) |
| `cosine_only` | yes | yes | full model |
| `dropout_variant` | yes | yes | plain Dropout replaces Spatial Dropout |
| `no_attention` | yes | no | multi-head self-attention removed |

All experiments use **K-Fold cross-validation**; results (per-fold weights and
`metrics.txt`) are written under the `results/` directory.

---

## Citation

If this work is helpful to your research, please cite:

```bibtex
@article{...}
```

## License

[To be filled in] -- MIT or Apache-2.0 is recommended.
