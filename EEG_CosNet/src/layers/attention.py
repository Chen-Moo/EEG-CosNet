# -*- coding: utf-8 -*-
"""
Spatial Dropout + Multi-Head Self-Attention composite module
(SpatialDropoutAttention).

This module encapsulates the second technical contribution of this work: after
each depthwise convolution block, apply **Spatial Dropout** (structured random
deactivation along the spatial dimension) first, then attach **Multi-Head
Self-Attention** (MHSA).

Design motivation
-----------------
1. Depthwise / separable convolutions have few parameters and limited capacity,
   and tend to overfit on individual channels or time segments during training.
   Spatial Dropout deactivates entire feature maps as a unit, which better
   matches the spatial correlation of convolutional feature maps than
   element-wise dropout, and effectively stabilizes training.
2. Multi-head self-attention allows the network to model long-range
   dependencies over the flattened feature sequence, compensating for the
   limited receptive field of local convolutions, while the multi-head
   mechanism lets different heads attend to different EEG time/channel patterns.

Implementation notes
--------------------
- ``spatial_attention_block``: a global-average-pooling based channel attention
  gate (Squeeze-and-Excitation style) for per-channel recalibration of feature maps.
- ``SpatialDropoutAttention``: wraps SpatialDropout2D and MultiHeadAttention
  into a reusable high-level module, applicable to any intermediate feature map.
"""

from keras.layers import (
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    Multiply,
    SpatialDropout2D,
    Dropout,
    MultiHeadAttention,
    Layer,
)


def spatial_attention_block(inputs):
    """Global-average-pooling based channel attention gate (channel-wise gating).

    Learns a per-channel softmax weight for the input feature map
    ``(B, H, W, C)`` and multiplies it element-wise with the original map,
    adaptively emphasizing more informative channels.

    Parameters
    ----------
    inputs : tensor
        Input feature map of shape ``(B, H, W, C)``.

    Returns
    -------
    tensor
        Feature map of the same shape, recalibrated by channel attention.
    """
    # Global average pooling: (B, H, W, C) -> (B, C)
    gap = GlobalAveragePooling2D()(inputs)
    # Reshape to (B, 1, 1, C)
    gap = Reshape((1, 1, -1))(gap)
    # Dense + softmax to produce attention weights
    attention = Dense(inputs.shape[-1], activation='softmax')(gap)
    # Reshape back to (B, H, W, C) for element-wise multiplication
    attention = Reshape(inputs.shape[1:])(attention)
    return Multiply()([inputs, attention])


class SpatialDropoutAttention(Layer):
    """Composite module of spatial dropout and multi-head self-attention.

    As a reusable Keras layer, it chains two stages of processing:

    1. **Structured dropout**: selects ``SpatialDropout2D`` or plain
       ``Dropout`` according to ``dropout_type`` to suppress overfitting and
       stabilize training.
    2. **Multi-head self-attention**: ``Reshape`` the feature map into a
       sequence ``(B, T, C)`` and apply ``MultiHeadAttention`` self-attention
       to capture long-range dependencies.

    Parameters
    ----------
    dropout_rate : float
        Dropout ratio.
    num_heads : int
        Number of attention heads.
    key_dim : int
        Dimension of query/key in each attention head.
    dropout_type : {'SpatialDropout2D', 'Dropout'}
        Dropout type, default ``'SpatialDropout2D'``.
    """

    def __init__(self,
                 dropout_rate=0.2,
                 num_heads=2,
                 key_dim=8,
                 dropout_type='SpatialDropout2D',
                 **kwargs):
        super(SpatialDropoutAttention, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_type = dropout_type

        if dropout_type == 'SpatialDropout2D':
            self._dropout = SpatialDropout2D(dropout_rate)
        elif dropout_type == 'Dropout':
            self._dropout = Dropout(dropout_rate)
        else:
            raise ValueError(
                "dropout_type must be one of 'SpatialDropout2D' or 'Dropout'.")

        self._mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs, training=None):
        """Forward pass: Dropout -> (Reshape) -> multi-head self-attention."""
        import tensorflow as tf

        x = self._dropout(inputs, training=training)

        # Feature map (B, H, W, C) -> sequence (B, H*W, C)
        shape = tf.shape(x)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        seq = tf.reshape(x, [b, h * w, c])

        # Multi-head self-attention (self-attention: q = k = v)
        attn = self._mha(seq, seq, seq, training=training)

        # Restore feature-map shape (B, H, W, C)
        return tf.reshape(attn, [b, h, w, c])

    def get_config(self):
        config = super(SpatialDropoutAttention, self).get_config()
        config.update({
            'dropout_rate': self.dropout_rate,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_type': self.dropout_type,
        })
        return config
