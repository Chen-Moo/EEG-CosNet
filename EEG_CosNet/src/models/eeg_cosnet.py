# -*- coding: utf-8 -*-
"""
EEG-CosNet: a cosine-convolution + attention network for EEG depression
recognition.

The model introduces two key technical improvements over the classical EEGNet:

1. **Cosine convolution (CosineConv2D)**: replaces the inner-product operation
   of a standard convolution with L2-normalization + cosine similarity, making
   feature extraction insensitive to EEG amplitude scaling and better matched
   to EEG characteristics.
2. **Spatial dropout + multi-head self-attention (SpatialDropoutAttention)**:
   inserts structured dropout and multi-head self-attention after each
   depthwise convolution block to stabilize training and model long-range
   dependencies.

This module provides two entry points, ``EEGNet`` and ``build_eeg_cosnet``.
The latter supports constructing the ablation variants via the
``use_cosine_conv`` and ``use_attention`` switches.

Dimension convention
--------------------
The depthwise convolution ``DepthwiseConv2D((chans, 1))`` compresses the
spatial dimension to ``1``, yielding a feature map ``(B, 1, T', C)``. The
multi-head self-attention operates on this 4D tensor (treating the time dim
``T'`` as the sequence length and ``C`` as the feature dim), so no extra
reshape is needed around attention; only the formal ``Reshape((-1, T', C))``
kept consistent with the original implementation is retained.
"""

from keras.models import Model
from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Conv2D,
    AveragePooling2D,
    SeparableConv2D,
    DepthwiseConv2D,
    BatchNormalization,
    SpatialDropout2D,
    Input,
    Flatten,
    Reshape,
    MultiHeadAttention,
)
from keras.constraints import max_norm

from src.layers import CosineConv2D


# ---------------------------------------------------------------------------
# Parameterized core model
# ---------------------------------------------------------------------------
def build_eeg_cosnet(nb_classes=2,
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
                     use_cosine_conv=True,
                     use_attention=True):
    """Build the EEG-CosNet model.

    Parameters
    ----------
    nb_classes : int
        Number of classes.
    chans : int
        Number of EEG channels.
    samples : int
        Number of time samples.
    dropout_rate : float
        Dropout ratio.
    kern_length : int
        Length of the first temporal convolution kernel (empirically half the
        sampling rate).
    f1 : int
        Number of temporal filters.
    depth_multiplier : int
        Spatial filter multiplier ``D`` of the depthwise convolution.
    f2 : int
        Number of pointwise filters output by the separable convolution.
    norm_rate : float
        Max-norm constraint on the classification layer weights.
    dropout_type : {'SpatialDropout2D', 'Dropout'}
        Dropout type.
    num_heads : int
        Number of attention heads.
    key_dim : int
        Dimension of each attention head.
    use_cosine_conv : bool
        Whether to use cosine convolution instead of standard convolution
        (ablation switch 1).
    use_attention : bool
        Whether to insert the multi-head self-attention module (ablation
        switch 2).

    Returns
    -------
    keras.Model
        An uncompiled Keras model instance.
    """
    if dropout_type == 'SpatialDropout2D':
        dropout_cls = SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropout_cls = Dropout
    else:
        raise ValueError(
            "dropout_type must be one of 'SpatialDropout2D' or 'Dropout'.")

    # Convolution factory: choose cosine or standard convolution per switch
    def _conv(filters, kernel_size, **kwargs):
        if use_cosine_conv:
            return CosineConv2D(filters, kernel_size, **kwargs)
        return Conv2D(filters, kernel_size, **kwargs)

    input_layer = Input(shape=(chans, samples, 1))

    # ===================== Block 1: temporal + depthwise spatial conv ======
    # First temporal convolution (along the time axis; cosine kernel by default)
    block1 = _conv(f1, (1, kern_length), padding='same',
                   use_bias=False)(input_layer)
    block1 = BatchNormalization()(block1)

    # Depthwise convolution: spatial filtering, compresses spatial dim to 1
    # -> (B, 1, samples, f1*D)
    block1 = DepthwiseConv2D((chans, 1),
                             use_bias=False,
                             depth_multiplier=depth_multiplier,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropout_cls(dropout_rate)(block1)

    # ---- Contribution 2: multi-head self-attention after spatial dropout ---
    if use_attention:
        # Formal reshape (consistent with the original implementation; the
        # dimensions are unchanged)
        block1 = Reshape((-1, (samples // 4), f1 * depth_multiplier))(block1)
        block1 = MultiHeadAttention(num_heads=num_heads,
                                    key_dim=key_dim)(block1, block1, block1)

    # ===================== Block 2: separable conv + attention ==============
    block2 = SeparableConv2D(f2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropout_cls(dropout_rate)(block2)

    if use_attention:
        block2 = Reshape((-1, (samples // 4 // 8), f2))(block2)
        block2 = MultiHeadAttention(num_heads=num_heads,
                                    key_dim=key_dim)(block2, block2, block2)

    # ===================== Classification head ==============================
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    output = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_layer, outputs=output)


# ---------------------------------------------------------------------------
# Backward-compatible entry: same signature as the original EEGNet
# ---------------------------------------------------------------------------
def EEGNet(nb_classes, Chans=7, Samples=1536,
           dropoutRate=0.2, kernLength=128, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='SpatialDropout2D'):
    """Backward-compatible model entry.

    Keeps the original parameter names (uppercase) and forwards to
    :func:`build_eeg_cosnet`, so existing call sites migrate without change.
    """
    return build_eeg_cosnet(
        nb_classes=nb_classes,
        chans=Chans,
        samples=Samples,
        dropout_rate=dropoutRate,
        kern_length=kernLength,
        f1=F1,
        depth_multiplier=D,
        f2=F2,
        norm_rate=norm_rate,
        dropout_type=dropoutType,
    )
